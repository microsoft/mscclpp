cd /root/mscclpp/python/mscclpp/language/tests/unit_tests
for file in *.py; do
  python3 "$file"
  if [ $? -ne 0 ]; then
    echo "❌ $file failed (exit code $?)"
  else
    echo "✅ $file succeeded"
  fi
done