set -e

mkdir -p /root/.ssh
mv /root/mscclpp/sshkey.pub /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
mv /root/mscclpp/config /root/.ssh/config
chown root:root /root/.ssh/config
chmod 400 /root/mscclpp/sshkey
chown root:root /root/mscclpp/sshkey

nvidia-smi -pm 1
for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
    nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i $i | sed 's/\ //') -i $i
done

pip3 install -r /root/mscclpp/python/test/requirements.txt

mkdir -p /var/run/sshd
/usr/sbin/sshd -p 22345
