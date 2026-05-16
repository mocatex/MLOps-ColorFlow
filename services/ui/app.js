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
    const debugToggle = document.getElementById("debug-color-toggle");
    const resetBtn = document.getElementById("reset-btn");

    let currentBlobUrl = null;
    let originalFileName = "image";
    let dragCounter = 0;

    let currentHighResFile = null;
    let currentLowResBlob = null;

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
        // Accept standard images OR HEIC
        if (!file.type.startsWith("image/") && !file.name.toLowerCase().endsWith(".heic")) {
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

        let processableFile = file;

        try {
            // 1. Intercept HEIC and convert to JPEG
            if (file.name.toLowerCase().endsWith(".heic") || file.type === "image/heic") {
                loadingSpinner.querySelector("p").textContent = "Converting Apple HEIC format...";
                const convertedBlob = await heic2any({
                    blob: file,
                    toType: "image/jpeg",
                    quality: 0.85,
                });
                // heic2any can return an array if it's an image sequence; grab the first one
                processableFile = Array.isArray(convertedBlob) ? convertedBlob[0] : convertedBlob;
            }

            // 2. Compress the image locally (Handles WebP, JPEG, PNG, and the converted HEIC natively)
            loadingSpinner.querySelector("p").textContent = "Compressing...";
            const safeBlob = await compressImage(processableFile);

            // Show the user the compressed version we are actually sending
            inputPreview.src = URL.createObjectURL(safeBlob);
            inputMeta.textContent = `${originalFileName} • ${(safeBlob.size / 1024).toFixed(1)} KB (Compressed)`;
            inputMeta.classList.remove("hidden");

            // 3. Send to server (Notice we pass BOTH the safe blob AND the high-res file now!)
            loadingSpinner.querySelector("p").textContent = "Processing via MLServer...";
            await processImage(safeBlob, processableFile);
        } catch (error) {
            console.error("Pipeline error:", error);
            alert(error.message);
            resetUI();
        }
    }

    // We now accept the highResFile as a parameter so the function doesn't crash!
    async function processImage(blobData, highResFile) {
        try {
            const body = new FormData();
            body.append("image", blobData);

            const response = await fetch(`${apiBaseUrl}/v2/models/colorflow/infer-image`, {
                method: "POST",
                body,
            });

            if (!response.ok) throw new Error("Backend inference failed.");

            const version = response.headers.get("X-Model-Version") || "unknown";
            currentLowResBlob = await response.blob();
            currentHighResFile = highResFile;

            // Run the trick!
            loadingSpinner.querySelector("p").textContent = "Enhancing resolution...";
            const highResBlob = await enhanceResolution(currentHighResFile, currentLowResBlob);

            currentBlobUrl = URL.createObjectURL(highResBlob);

            outputPreview.src = currentBlobUrl;
            outputMeta.textContent = `PNG • MLflow Champion v${version} (Enhanced)`;

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

    /**
     * Merges the sharp details (Luminance) of the original high-res image
     * with the colors of the GAN's low-res output using luminosity blending.
     */
    async function enhanceResolution(originalFile, lowResColorBlob) {
        return new Promise((resolve, reject) => {
            const origImg = new Image();
            const colorImg = new Image();

            let loaded = 0;
            const checkReady = () => {
                loaded++;
                if (loaded === 2) performMerge();
            };

            origImg.onload = checkReady;
            colorImg.onload = checkReady;
            origImg.onerror = () => reject(new Error("Failed to load original image for merging."));
            colorImg.onerror = () => reject(new Error("Failed to load color image for merging."));

            origImg.src = URL.createObjectURL(originalFile);
            colorImg.src = URL.createObjectURL(lowResColorBlob);

            function performMerge() {
                const w = origImg.width,
                    h = origImg.height;

                // Canvas for high-res original
                const origCanvas = document.createElement("canvas");
                origCanvas.width = w;
                origCanvas.height = h;
                const origCtx = origCanvas.getContext("2d");
                origCtx.drawImage(origImg, 0, 0);
                const origData = origCtx.getImageData(0, 0, w, h).data;

                // Canvas for low-res color image (stretched)
                const colorCanvas = document.createElement("canvas");
                colorCanvas.width = w;
                colorCanvas.height = h;
                const colorCtx = colorCanvas.getContext("2d");
                colorCtx.drawImage(colorImg, 0, 0, w, h);
                const colorData = colorCtx.getImageData(0, 0, w, h).data;

                // Output canvas
                const outCanvas = document.createElement("canvas");
                outCanvas.width = w;
                outCanvas.height = h;
                const outCtx = outCanvas.getContext("2d");
                const outImageData = outCtx.createImageData(w, h);
                const outPixels = outImageData.data;

                const isDebugMode = document.getElementById("debug-color-toggle").checked;

                for (let i = 0; i < outPixels.length; i += 4) {
                    // Luminance from high-res original (ITU-R BT.709)
                    let L;

                    if (isDebugMode) {
                        // For debug mode, force flat 50% gray to isolate pure chrominance
                        L = 128;
                    } else {
                        // Normal mode: extract sharp luminance from the original image
                        L = 0.2126 * origData[i] + 0.7152 * origData[i + 1] + 0.0722 * origData[i + 2];
                    }

                    // Color from the low-res GAN output
                    const Rc = colorData[i];
                    const Gc = colorData[i + 1];
                    const Bc = colorData[i + 2];

                    // Convert GAN color to HSL, replace its Lightness with our chosen L
                    const [h, s, _] = rgbToHsl(Rc, Gc, Bc);
                    const [r, g, b] = hslToRgb(h, s, L / 255);

                    outPixels[i] = r;
                    outPixels[i + 1] = g;
                    outPixels[i + 2] = b;
                    outPixels[i + 3] = 255; // Alpha
                }

                outCtx.putImageData(outImageData, 0, 0);

                outCanvas.toBlob((blob) => {
                    URL.revokeObjectURL(origImg.src);
                    URL.revokeObjectURL(colorImg.src);
                    if (blob) resolve(blob);
                    else reject(new Error("Canvas export failed"));
                }, "image/png");
            }
        });
    }

    function rgbToHsl(r, g, b) {
        r /= 255;
        g /= 255;
        b /= 255;
        const max = Math.max(r, g, b),
            min = Math.min(r, g, b);
        let h,
            s,
            l = (max + min) / 2;
        if (max === min) {
            h = s = 0;
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r:
                    h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
                    break;
                case g:
                    h = ((b - r) / d + 2) / 6;
                    break;
                case b:
                    h = ((r - g) / d + 4) / 6;
                    break;
            }
        }
        return [h, s, l];
    }

    function hslToRgb(h, s, l) {
        let r, g, b;
        if (s === 0) {
            r = g = b = l;
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1 / 6) return p + (q - p) * 6 * t;
                if (t < 1 / 2) return q;
                if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
                return p;
            };
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1 / 3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1 / 3);
        }
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
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

    debugToggle.addEventListener("change", async () => {
        if (!currentHighResFile || !currentLowResBlob) return;

        loadingSpinner.querySelector("p").textContent = "Recalculating canvas...";
        loadingSpinner.classList.remove("hidden");
        outputPreview.classList.add("hidden");

        try {
            const highResBlob = await enhanceResolution(currentHighResFile, currentLowResBlob);

            if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
            currentBlobUrl = URL.createObjectURL(highResBlob);
            outputPreview.src = currentBlobUrl;
        } catch (error) {
            console.error(error);
        } finally {
            loadingSpinner.classList.add("hidden");
            outputPreview.classList.remove("hidden");
        }
    });

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
