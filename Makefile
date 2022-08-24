# copied from:
# https://spin.atomicobject.com/2016/08/26/makefile-c-projects/
#
# Big thanks for Job Varnish <3
#

TARGET_EXEC ?= a.out

TEST_GENERATOR ?= generate_functional_tests.py

BUILD_DIR ?= ./build
SRC_DIRS ?= ./sources
INC_DIRS ?= ./include/MEGANI

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(INC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

OPTIMIZE ?= -O2
CFLAGS ?= -g $(OPTIMIZE) -std=c99 -pedantic-errors -Werror -Wall -Wfatal-errors -Wextra $(INC_FLAGS) -MMD -MP

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS) -lm

test:
	python3 scripts/$(TEST_GENERATOR)
	make
	$(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)
	rm -rf $(SRC_DIRS)/main.c

-include $(DEPS)

MKDIR_P ?= mkdir -p
