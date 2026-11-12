#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -euxo pipefail

MOUNT=/mnt/nvme
LABEL=mscclpp_nvme
exec 9>/run/lock/mscclpp-nvme.lock
flock 9

test ! -L "$MOUNT"
mkdir -p "$MOUNT"
if ! mountpoint -q "$MOUNT"; then
    test -z "$(ls -A "$MOUNT")"
fi

DISK=$(blkid -L "$LABEL" || true)
if [ -z "$DISK" ]; then
    if findmnt --fstab --mountpoint "$MOUNT" >/dev/null || mountpoint -q "$MOUNT"; then
        echo "Configured NVMe is unavailable; refusing to format another disk." >&2
        exit 1
    fi
    for candidate in $(lsblk -bdnpo NAME,SIZE,TYPE | awk '$1 ~ /^\/dev\/nvme[0-9]+n[0-9]+$/ && $2 >= 3298534883328 && $3 == "disk" {print $1}'); do
        [ "$(lsblk -nrpo NAME "$candidate" | wc -l)" -eq 1 ] || continue
        [ -z "$(lsblk -nro FSTYPE,MOUNTPOINTS "$candidate" | tr -d '[:space:]')" ] || continue
        [ -z "$(wipefs --no-act --noheadings "$candidate")" ] || continue
        fuser -s "$candidate" && continue
        DISK=$candidate
        break
    done
    test -n "$DISK"
    mkfs.xfs -L "$LABEL" "$DISK"
fi

UUID=$(blkid -s UUID -o value "$DISK")
ENTRY="UUID=$UUID $MOUNT xfs defaults,nofail 0 0"
if ! grep -Fxq "$ENTRY" /etc/fstab; then
    test -z "$(findmnt --fstab -nr --mountpoint "$MOUNT")"
    echo "$ENTRY" >> /etc/fstab
fi
mountpoint -q "$MOUNT" || mount "$MOUNT"
test "$(findmnt -nro UUID --mountpoint "$MOUNT")" = "$UUID"

READY=true
for service in docker containerd; do
    mkdir -p "$MOUNT/$service" "/var/lib/$service"
    if mountpoint -q "/var/lib/$service"; then
        test "$MOUNT/$service" -ef "/var/lib/$service"
    else
        READY=false
    fi
done
if [ "$READY" = false ]; then
    if systemctl is-active --quiet docker; then
        test "$(docker info --format '{{.DockerRootDir}}')" = /var/lib/docker
        test -z "$(docker ps -q)"
    fi
    if systemctl is-active --quiet containerd; then
        for namespace in $(ctr namespaces list -q); do
            test -z "$(ctr --namespace "$namespace" tasks list -q)"
        done
    fi
    systemctl stop docker.service docker.socket containerd.service
fi

for service in docker containerd; do
    mkdir -p "/etc/systemd/system/$service.service.d"
    cat > "/etc/systemd/system/$service.service.d/mscclpp-nvme.conf" <<EOF
[Unit]
RequiresMountsFor=$MOUNT /var/lib/docker /var/lib/containerd
AssertPathIsMountPoint=$MOUNT
AssertPathIsMountPoint=/var/lib/docker
AssertPathIsMountPoint=/var/lib/containerd
EOF
    TARGET="/var/lib/$service"
    ENTRY="$MOUNT/$service $TARGET none bind,nofail,x-systemd.requires-mounts-for=$MOUNT 0 0"
    if ! grep -Fxq "$ENTRY" /etc/fstab; then
        test -z "$(findmnt --fstab -nr --mountpoint "$TARGET")"
        echo "$ENTRY" >> /etc/fstab
    fi
done
systemctl daemon-reload
for service in docker containerd; do
    mountpoint -q "/var/lib/$service" || mount "/var/lib/$service"
    test "$MOUNT/$service" -ef "/var/lib/$service"
done
systemctl start containerd.service docker.socket docker.service
docker info --format '{{.DockerRootDir}}'
