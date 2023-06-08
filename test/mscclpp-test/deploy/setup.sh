set -e

mv /root/mscclpp/sshkey.pub /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
mv /root/mscclpp/config /root/.ssh/config
chown root:root /root/.ssh/config
chmod 400 /root/mscclpp/sshkey
chown root:root /root/mscclpp/sshkey

/usr/sbin/sshd -p 22345
