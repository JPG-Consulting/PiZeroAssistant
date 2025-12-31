# Simple FTP setup on Raspberry Pi Zero 2 W (beginner-friendly)

This guide explains, **step by step**, how to set up a **very simple FTP server** on a Raspberry Pi Zero 2 W.

It is intended **only for local development** (for example: uploading/downloading source code and audio files for a voice assistant).

- ❌ Not for production
- ❌ Not exposed to the internet
- ✅ Local network only
- ✅ Minimal configuration
- ✅ Uses your **existing Raspberry Pi user**

The FTP **CHROOT** (root folder when connected) will be:

```
/home/<your-username>/ftp
```

Uploaded and downloaded files will live in:

```
/home/<your-username>/ftp/uploads
```

---

## What you will need

- Raspberry Pi Zero 2 W with Raspberry Pi OS installed
- The Pi connected to your **home network**
- A keyboard + screen **or** SSH access
- Your normal Raspberry Pi username (for example: `pi`)
- About 10 minutes ⏱️

---

## Step 1 – Update the Raspberry Pi

Open a terminal on the Raspberry Pi and type:

```bash
sudo apt update
sudo apt upgrade -y
```

---

## Step 2 – Install the FTP server

We will use **vsftpd**, a lightweight and simple FTP server.

```bash
sudo apt install -y vsftpd
```

---

## Step 3 – Create the FTP folders (CHROOT directory)

We will create an `ftp` folder **inside your home directory**, with an `uploads` subfolder.

⚠️ Replace `YOUR_USERNAME` with your actual Raspberry Pi username  
(example: `pi`)

```bash
mkdir /home/YOUR_USERNAME/ftp
mkdir /home/YOUR_USERNAME/ftp/uploads
```

📁 Files you upload and download will be located in:

```
/home/YOUR_USERNAME/ftp/uploads
```

---

## Step 4 – Configure vsftpd (minimal setup)

### Back up the original configuration

```bash
sudo cp /etc/vsftpd.conf /etc/vsftpd.conf.backup
```

### Edit the configuration file

```bash
sudo nano /etc/vsftpd.conf
```

### Replace the content with this minimal configuration

```conf
listen=YES
listen_ipv6=NO

local_enable=YES
write_enable=YES

chroot_local_user=YES
allow_writeable_chroot=YES
local_root=/home/$USER/ftp
```

Save and exit:
- Press **CTRL + O**, then **Enter**
- Press **CTRL + X**

---

## Step 5 – Restart the FTP service

```bash
sudo systemctl restart vsftpd
sudo systemctl enable vsftpd
```

Check that it is running:

```bash
sudo systemctl status vsftpd
```

You should see:

```
active (running)
```

---

## Step 6 – Find the Raspberry Pi IP address

```bash
hostname -I
```

Example:

```
192.168.1.50
```

---

## Step 7 – Connect from your computer

### Option A – FileZilla (recommended)

1. Install **FileZilla**
2. Open it and enter:
   - **Host:** `ftp://<PI_IP>`
   - **Username:** your Raspberry Pi username
   - **Password:** your Raspberry Pi password
   - **Port:** `21`
3. Click **Quickconnect**

You will see the `uploads` folder.

---

## Step 8 – Test upload & download

1. Open the `uploads` folder
2. Upload a small file (source code or audio)
3. Download it back
4. ✅ Done

---

## Important notes (keep it simple)

- This FTP server is **local only**
- Do **not** open ports on your router
- Do **not** use this on the public internet
- Passwords are sent **unencrypted** (OK for local use only)
- If you want security later, use **SFTP instead**

---

## Where your files are stored

```
/home/<your-username>/ftp/uploads
```

You can open this folder directly on the Raspberry Pi.

---

## That’s it 🎉

You now have a **simple FTP server** using your existing Raspberry Pi user, perfect for quickly transferring code and audio files during development.
