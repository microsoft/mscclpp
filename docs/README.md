## How to build docs

1. Install `doxygen`.

    ```bash
    $ sudo apt-get install doxygen graphviz
    ```

2. Install Python packages below. If you install them on the user's local, you need to include `~/.local/bin` to `$PATH` (to use `sphinx-build`).

    ```bash
    $ sudo python3 -m pip install -r ./requirements.txt
    ```

3. Create Doxygen documents.

    ```bash
    $ doxygen
    ```

4. Create Sphinx documents.

    ```bash
    $ make html
    ```

5. Done. The HTML files will be on `_build/` directory.
