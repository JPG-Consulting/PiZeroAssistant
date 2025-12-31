# Install Respeaker 2-mic array hat

This guide has been written to install a **KEYESTUDIO ReSpeaker 2-Mic Pi Hat (V1)** but should work for any **ReSpeaker** hat based on **WM8960** codec.

## 1. Check your kernel version

To check your kernel version run:

```bash
uname -r
```

The output will be something like

```
6.12.47+rpt-rpi-v8
```

In this example we are interested in the `6.12` as it will be the branch we shall checkout in the following step.

## 2. Install the driver

We will have to run

```bash
git clone https://github.com/HinTak/seeed-voicecard
cd seeed-voicecard
git checkout v6.12
sudo ./install.sh
sudo reboot
```

**IMPORTANT:** Please note that the command `git checkout v6.12` is related to the kernel version we have gotten on the previous step. If your kernel version is different you should use your own kernel version.

## 3. Test

To test the driver installation we will run:

```bash
aplay -l
```

It should rturn something like:

``` 
**** List of PLAYBACK Hardware Devices ****
card 0: vc4hdmi [vc4-hdmi], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: seeed2micvoicec [seeed-2mic-voicecard], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0 [bcm2835-i2s-wm8960-hifi wm8960-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

```

Where we can see our hat has been installed as `card 1`.

Then we can run:

```bash
arecord -l
```

This command should output something like:

```
**** List of CAPTURE Hardware Devices ****
card 1: seeed2micvoicec [seeed-2mic-voicecard], device 0: bcm2835-i2s-wm8960-hifi wm8960-hifi-0 [bcm2835-i2s-wm8960-hifi wm8960-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

```

Where we can see our hat has been installed as `card 1`.

Now it is the time to make a recording:

```bash
arecord -D hw:1,0 -f S16_LE -c 2 -r 44100 -d 5 -t wav test.wav
```

...and play it back with:

```bash
 aplay -D hw:1,0 test.wav
```

**NOTE:** we are using `hw:1,0` because our hat has installed as `card 1` and subdevice `#0`.


