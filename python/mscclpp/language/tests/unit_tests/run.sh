for file in *.py; do
  python3 "$file"
done

echo "Exit code: $?"