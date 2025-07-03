cd /root/mscclpp/python/mscclpp/language/tests
set -e  
# Recursively find all .py files and run them one by one:
find . -type f -name '*.py' -print0 | \
  while IFS= read -r -d '' file; do
    echo "→ Running $file"
    python3 "$file"
    code=$?
    if [ $code -ne 0 ]; then
      echo "❌ $file failed with exit code $code"
    else
      echo "✅ $file succeeded"
    fi
  done