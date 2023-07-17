import mscclpp

def main():
    config = mscclpp.Config.get_instance()
    config.set_bootstrap_connection_timeout_config(15)
    timeout = config.get_bootstrap_connection_timeout_config()
    assert timeout == 15

if __name__ == "__main__":
    main()
