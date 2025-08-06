TUTORIAL_DIRS := 01-basic-concepts 02-bootstrap 03-memory-channel

.PHONY: all clean help $(TUTORIAL_DIRS) clean-%

all: $(TUTORIAL_DIRS)
	@echo "All tutorials built successfully!"

$(TUTORIAL_DIRS):
	@echo "Building tutorial: $@"
	@$(MAKE) -C $@ all

clean: $(addprefix clean-, $(TUTORIAL_DIRS))
	@echo "All tutorials cleaned!"

clean-%:
	@echo "Cleaning tutorial: $*"
	@$(MAKE) -C $* clean
