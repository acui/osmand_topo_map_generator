#!/bin/bash
# Reimplementation of the legacy OsmAnd externals helper functions
# (prepareUpstream*/patchUpstream). State is tracked via the "stamp" file.

function _download {
	local from=$1
	local to=$2
	echo "Downloading $from to $to"
	local i=0
	local exitCode=1
	while [ $i -lt 3 ]; do
		curl -L --fail --retry 3 --connect-timeout 30 "$from" > "$to" && exitCode=0 && break
		i=$((i+1))
		sleep 5
	done
	if [ $exitCode -ne 0 ]; then
		echo "Failed to download $from" >&2
		rm -f "$to"
	fi
	return $exitCode
}

function _stampValue {
	local dir=$1
	if [ -f "$dir/stamp" ]; then
		cat "$dir/stamp"
	else
		echo ""
	fi
}

function _markStamp {
	local dir=$1
	local value=$2
	echo "$value" > "$dir/stamp"
}

function prepareUpstreamFromTarArchive {
	local dir=$1
	local url=$2
	local stamp=$(_stampValue "$dir")
	if [ -d "$dir/upstream" ] && [ -n "$(ls -A "$dir/upstream")" ] && [ "$stamp" != "#0" ] && [ -n "$stamp" ]; then
		echo "$dir: upstream already prepared ($stamp)"
		return 0
	fi
	rm -rf "$dir/upstream"
	mkdir -p "$dir/upstream" "$dir/_archive"
	local archiveName=$(basename "$url")
	local archivePath="$dir/_archive/$archiveName"
	_download "$url" "$archivePath" || return 1
	local tmpExtract="$dir/_archive/extract"
	rm -rf "$tmpExtract" && mkdir -p "$tmpExtract"
	case "$archiveName" in
		*.tar.bz2|*.tbz2) tar -xjf "$archivePath" -C "$tmpExtract" ;;
		*.tar.gz|*.tgz)   tar -xzf "$archivePath" -C "$tmpExtract" ;;
		*.tar.xz)         tar -xJf "$archivePath" -C "$tmpExtract" ;;
		*) echo "Unknown archive type: $archiveName" >&2; return 1 ;;
	esac
	# flatten the single top-level directory if present
	local entries=$(ls -A "$tmpExtract" | wc -l)
	if [ "$entries" = "1" ] && [ -d "$tmpExtract/$(ls -A "$tmpExtract")" ]; then
		mv "$tmpExtract/$(ls -A "$tmpExtract")"/* "$dir/upstream"/
	else
		mv "$tmpExtract"/* "$dir/upstream"/ 2>/dev/null || mv "$tmpExtract"/.[!.]* "$dir/upstream"/ 2>/dev/null
	fi
	rm -rf "$tmpExtract"
	_markStamp "$dir" "#tar:$archiveName"
	echo "$dir: upstream prepared from tar archive"
}

function prepareUpstreamFromZipArchive {
	local dir=$1
	local url=$2
	local stamp=$(_stampValue "$dir")
	if [ -d "$dir/upstream" ] && [ -n "$(ls -A "$dir/upstream")" ] && [ "$stamp" != "#0" ] && [ -n "$stamp" ]; then
		echo "$dir: upstream already prepared ($stamp)"
		return 0
	fi
	rm -rf "$dir/upstream"
	mkdir -p "$dir/upstream" "$dir/_archive"
	local archiveName=$(basename "$url")
	local archivePath="$dir/_archive/$archiveName"
	_download "$url" "$archivePath" || return 1
	local tmpExtract="$dir/_archive/extract"
	rm -rf "$tmpExtract" && mkdir -p "$tmpExtract"
	unzip -q -o "$archivePath" -d "$tmpExtract"
	local entries=$(ls -A "$tmpExtract" | wc -l)
	if [ "$entries" = "1" ] && [ -d "$tmpExtract/$(ls -A "$tmpExtract")" ]; then
		mv "$tmpExtract/$(ls -A "$tmpExtract")"/* "$dir/upstream"/
	else
		mv "$tmpExtract"/* "$dir/upstream"/
	fi
	rm -rf "$tmpExtract"
	_markStamp "$dir" "#zip:$archiveName"
	echo "$dir: upstream prepared from zip archive"
}

function prepareUpstreamFromGit {
	local dir=$1
	local repo=$2
	local ref=$3
	local stamp=$(_stampValue "$dir")
	if [ -d "$dir/upstream" ] && [ -n "$(ls -A "$dir/upstream")" ] && [ "$stamp" != "#0" ] && [ -n "$stamp" ]; then
		echo "$dir: upstream already prepared ($stamp)"
		return 0
	fi
	rm -rf "$dir/upstream"
	if ! git clone --depth 1 -b "$ref" "$repo" "$dir/upstream" 2>/dev/null; then
		git clone "$repo" "$dir/upstream" || return 1
		git -C "$dir/upstream" checkout "$ref" || return 1
	fi
	_markStamp "$dir" "#git:$ref"
	echo "$dir: upstream prepared from git ($repo @ $ref)"
}

function patchUpstream {
	local dir=$1
	if [ ! -d "$dir/upstream" ]; then
		echo "$dir: no upstream to patch" >&2
		return 1
	fi
	rm -rf "$dir/upstream.patched"
	cp -R "$dir/upstream" "$dir/upstream.patched"
	if [ -d "$dir/patches" ]; then
		local patchScript
		for patchScript in $(ls "$dir/patches/"*.sh 2>/dev/null | sort); do
			echo "$dir: running $(basename "$patchScript")"
			(cd "$dir/upstream.patched" && bash "$patchScript") || return 1
		done
		local patchFile
		for patchFile in $(ls "$dir/patches/"*.patch 2>/dev/null | sort); do
			echo "$dir: applying $(basename "$patchFile")"
			patch --silent -p1 -d "$dir/upstream.patched" -i "$patchFile" || return 1
		done
	fi
	_markStamp "$dir" "$(_stampValue "$dir")-patched"
	echo "$dir: upstream patched"
}
