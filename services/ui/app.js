document.addEventListener("DOMContentLoaded", () => {
    // --- API & Config ---
    const apiBaseUrl = (() => {
        if (window.COLORFLOW_API_BASE_URL) return window.COLORFLOW_API_BASE_URL.replace(/\/$/, "");
        const isLocalhost = ["localhost", "127.0.0.1"].includes(window.location.hostname);
        if (isLocalhost && window.location.port && window.location.port !== "8080") {
            return `${window.location.protocol}//127.0.0.1:8080`;
        }
        return "";
    })();

    // Security Limits
    const MAX_DIMENSION = 2048; // Max width or height
    const MAX_FILE_SIZE_MB = 4; // Max size after compression

    // --- DOM Elements ---
    const dropOverlay = document.getElementById("fullscreen-drop-overlay");
    const uploadPrompt = document.getElementById("upload-prompt");
    const workspace = document.getElementById("workspace");
    const fileInput = document.getElementById("file-input");

    const inputPreview = document.getElementById("input-preview");
    const outputPreview = document.getElementById("output-preview");
    const inputMeta = document.getElementById("input-meta");
    const outputMeta = document.getElementById("output-meta");
    const loadingSpinner = document.getElementById("loading-spinner");

    const downloadBtn = document.getElementById("download-btn");
    const resetBtn = document.getElementById("reset-btn");

    let currentBlobUrl = null;
    let originalFileName = "image";
    let dragCounter = 0;

    // --- Global Drag & Drop Handlers ---
    document.body.addEventListener("dragenter", (e) => {
        e.preventDefault();
        dragCounter++;
        dropOverlay.classList.add("active");
    });

    document.body.addEventListener("dragover", (e) => {
        e.preventDefault();
    });

    document.body.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter === 0) dropOverlay.classList.remove("active");
    });

    document.body.addEventListener("drop", (e) => {
        e.preventDefault();
        dragCounter = 0;
        dropOverlay.classList.remove("active");
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // --- Client-Side Compression ---
    async function compressImage(file) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                let width = img.width;
                let height = img.height;

                // 1. Scale down if it exceeds max dimensions
                if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
                    if (width > height) {
                        height = Math.round((height * MAX_DIMENSION) / width);
                        width = MAX_DIMENSION;
                    } else {
                        width = Math.round((width * MAX_DIMENSION) / height);
                        height = MAX_DIMENSION;
                    }
                }

                // 2. Draw to canvas (This naturally flattens to 8-bit RGB)
                const canvas = document.createElement("canvas");
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0, width, height);

                // 3. Export as compressed JPEG
                canvas.toBlob(
                    (blob) => {
                        URL.revokeObjectURL(img.src);
                        if (!blob) return reject(new Error("Canvas export failed"));

                        // 4. Hard limit check to protect the server wallet
                        const sizeMB = blob.size / (1024 * 1024);
                        if (sizeMB > MAX_FILE_SIZE_MB) {
                            reject(
                                new Error(
                                    `Image is too complex. Even after compression it is ${sizeMB.toFixed(1)}MB (Limit is ${MAX_FILE_SIZE_MB}MB).`,
                                ),
                            );
                        } else {
                            resolve(blob);
                        }
                    },
                    "image/jpeg",
                    0.85,
                ); // 85% quality is a sweet spot for ML inputs
            };
            img.onerror = () => reject(new Error("Failed to load image for compression."));
            img.src = URL.createObjectURL(file);
        });
    }

    // --- Processing Logic ---
    async function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file.");
            return;
        }

        originalFileName = file.name;

        // Immediately transition UI to the loading state
        uploadPrompt.classList.add("hidden");
        workspace.classList.remove("hidden");
        outputPreview.classList.add("hidden");
        outputMeta.classList.add("hidden");
        loadingSpinner.classList.remove("hidden");
        downloadBtn.disabled = true;

        if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);

        try {
            // Compress the image locally first!
            const safeBlob = await compressImage(file);

            // Show the user the compressed version we are actually sending
            inputPreview.src = URL.createObjectURL(safeBlob);
            inputMeta.textContent = `${originalFileName} • ${(safeBlob.size / 1024).toFixed(1)} KB (Compressed)`;
            inputMeta.classList.remove("hidden");

            // Send to server
            await processImage(safeBlob);
        } catch (error) {
            console.error("Compression error:", error);
            alert(error.message);
            resetUI();
        }
    }

    async function processImage(blobData) {
        try {
            const body = new FormData();
            body.append("image", blobData);

            const response = await fetch(`${apiBaseUrl}/v2/models/colorflow/infer-image`, {
                method: "POST",
                body,
            });

            if (!response.ok) throw new Error("Backend inference failed.");

            const version = response.headers.get("X-Model-Version") || "unknown";

            const resultBlob = await response.blob();
            currentBlobUrl = URL.createObjectURL(resultBlob);

            outputPreview.src = currentBlobUrl;
            outputMeta.textContent = `PNG • MLflow Champion v${version}`;

            outputPreview.classList.remove("hidden");
            outputMeta.classList.remove("hidden");
            loadingSpinner.classList.add("hidden");
            downloadBtn.disabled = false;

            // 🎉 FIRE THE CONFETTI! 🎉
            if (typeof confetti === "function") {
                confetti({
                    particleCount: 150,
                    spread: 80,
                    origin: { y: 0.6 },
                    colors: ["#ff6b00", "#ffb347", "#ffffff", "#2a2a2a"],
                });
            }
        } catch (error) {
            console.error("Colorization error:", error);
            alert("Failed to colorize image. Check if MLServer is running.");
            resetUI();
        }
    }

    // --- Actions ---
    downloadBtn.addEventListener("click", () => {
        if (!currentBlobUrl) return;
        const baseName = originalFileName.replace(/\.[^/.]+$/, "");
        const newFileName = `${baseName}-colorized.png`;

        const a = document.createElement("a");
        a.href = currentBlobUrl;
        a.download = newFileName;
        a.click();
    });

    resetBtn.addEventListener("click", resetUI);

    function resetUI() {
        workspace.classList.add("hidden");
        uploadPrompt.classList.remove("hidden");
        inputPreview.src = "";
        outputPreview.src = "";
        inputMeta.textContent = "";
        outputMeta.textContent = "";
        fileInput.value = "";
        if (currentBlobUrl) {
            URL.revokeObjectURL(currentBlobUrl);
            currentBlobUrl = null;
        }
    }
});
