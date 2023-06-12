set -e

mkdir -p /root/.ssh
mv /root/mscclpp/sshkey.pub /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
mv /root/mscclpp/config /root/.ssh/config
chown root:root /root/.ssh/config
chmod 400 /root/mscclpp/sshkey
chown root:root /root/mscclpp/sshkey

apt-get update -y
apt-get install openssh-server -y

mkdir -p /var/run/sshd
/usr/sbin/sshd -p 22345
