#!/bin/bash

set -eu

TEST_DATA_URL=${TEST_DATA_URL:-https://big-tank.app.tu-dortmund.de/lst-testdata/}
TEST_DATA_USER=${TEST_DATA_USER:-""}
TEST_DATA_PASSWORD=${TEST_DATA_PASSWORD:-""}


if [ -z "$TEST_DATA_USER" ]; then
	echo -n "Username: "
	read TEST_DATA_USER
	echo
fi

if [ -z "$TEST_DATA_PASSWORD" ]; then
	echo -n "Password: "
	read -s TEST_DATA_PASSWORD
	echo
fi


wget \
	-R "*.html*,*.gif" \
	--no-host-directories --cut-dirs=1 \
	--no-parent \
	--level=inf \
	--user="$TEST_DATA_USER" \
	--password="$TEST_DATA_PASSWORD" \
	--no-verbose \
	--recursive \
	--directory-prefix=test_data \
	"$TEST_DATA_URL"
