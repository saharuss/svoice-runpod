# svoice Web App

A Next.js application for separating speaker voices from mixed audio using RunPod Serverless.

## Setup

1.  **Install Dependencies**:
    ```bash
    npm install
    ```

2.  **Configure Environment Variables**:
    Create a `.env.local` file in the `svoice-web` directory:
    ```env
    RUNPOD_API_KEY=your_runpod_api_key_here
    RUNPOD_ENDPOINT_ID=your_endpoint_id_here
    ```

    > **Note**: You can get your API Key from [RunPod Settings](https://www.runpod.io/console/user/settings) and Endpoint ID from [Serverless Console](https://www.runpod.io/console/serverless).

3.  **Run Development Server**:
    ```bash
    npm run dev
    ```
    Open [http://localhost:3000](http://localhost:3000) in your browser.

## Features
- Drag & Drop Audio Upload
- Real-time Processing Status
- Secure Server-side API Proxy
- Interactive Audio Players for Separated Tracks
- Premium Dark Mode UI
