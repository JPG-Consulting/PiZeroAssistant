#!/usr/bin/env bash
set -e

CARD=1

echo "[ALSA] Restoring ReSpeaker mic settings on card $CARD"

# Capture path
amixer -c $CARD cset numid=3 on,on            # Capture Switch
amixer -c $CARD cset numid=1 63,63             # Capture Volume
amixer -c $CARD cset numid=36 200,200           # ADC PCM Capture Volume

# Input mixer
amixer -c $CARD cset numid=49 on                # Left Input Mixer Boost
amixer -c $CARD cset numid=50 on                # Right Input Mixer Boost

# Use LINPUT2 / RINPUT2
amixer -c $CARD cset numid=5 7                  # LINPUT2 boost
amixer -c $CARD cset numid=7 7                  # RINPUT2 boost

# Disable ALC (critical)
amixer -c $CARD cset numid=26 0                 # ALC Off

echo "[ALSA] ReSpeaker mic configuration applied"
