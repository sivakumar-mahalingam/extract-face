# Extract Face
Extracts face image from ID cards and stores it

## **Setup Instructions**

### **1. Install Tesseract OCR**
1. Download the Tesseract OCR installer from the link below:
   - [Tesseract OCR 5.5.0 for Windows](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)
2. Run the installer and follow the instructions to complete the installation.
   - **Important:** During installation, ensure you check the option to add Tesseract to the system PATH.
3. After installation, verify that Tesseract is installed correctly by running the following command in the **Command Prompt**:
   ```bash
   tesseract --version
   ```
   You should see the version details printed.

### **2. Clone the Python Repository**
1. Navigate to the repository directory:
   ```bash
   cd <repository-directory>
   ```

### **3. Set Up PyCharm**
1. Open **PyCharm** and select **Open** to open the repository.
2. Set up a **virtual environment**:
   - Go to **File > Settings > Project > Python Interpreter**.
   - Click on the gear icon and select **Add Interpreter > Add Local Interpreter**.
   - Choose **Virtualenv** and click **OK** to create and configure the virtual environment.
3. Activate the virtual environment:
   - Open the **Terminal** within PyCharm and run:
     ```bash
     venv\Scripts\activate
     ```

### **4. Install Dependencies**
1. Make sure `pip` is updated:
   ```bash
   python -m pip install --upgrade pip
   ```
2. Install all the required packages listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

#### **5. Run Your Code**
1. To test if everything is set up correctly, try running a script in your repository (for example, a test script or the main application script).
   ```bash
   python <test_script>.py
   ```
