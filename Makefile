#tools
shell = /bin/bash
MKDIR_P ?= mkdir -p

#directories
BUILD_DIRS ?= ./build
SRC_DIRS ?= ./sources
INC_DIRS := $(shell find $(SRC_DIRS) -type d)

#flags
INC_FLAGS := $(addprefix -I,$(INC_DIRS))
CFLAGS ?= -g -O0 -pedantic-errors -Werror -Wall -Wfatal-errors -Wextra $(INC_FLAGS)

#files
SRCS := $(shell find $(SRC_DIRS) -name *.c)
OBJS := $(SRCS:%=$(BUILD_DIRS)/%.o)

TARGET_EXEC ?= a.out

#all
all:$(BUILD_DIRS)/$(TARGET_EXEC)

$(BUILD_DIRS)/$(TARGET_EXEC): $(OBJS)
	echo $(INC_FLAGS)
	$(CC) $(CFLAGS) $(OBJS) -o $@

#C89 lang
$(BUILD_DIRS)/%.ansi.c.o: %.ansi.c
	$(MKDIR_P) $(dir $@)
	$(CC) -ansi $(CFLAGS) -c $^ -o $@

#C17 lang
$(BUILD_DIRS)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) -std=c17 $(CFLAGS) -c $^ -o $@

#clean
clean:
	rm -r $(BUILD_DIRS)/*