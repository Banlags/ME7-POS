document.addEventListener("DOMContentLoaded", () => {
    const SESSION_STATUS_ELEMENT = document.getElementById("session-status");
    const RECEIPT_BODY_ELEMENT = document.getElementById("receipt-body");
    const TOTAL_VALUE_ELEMENT = document.getElementById("total-value");
    const SESSION_SUMMARY_ELEMENT = document.getElementById("session-summary");

    const soundItem = document.getElementById("sound-item");
    const soundStart = document.getElementById("sound-start");
    const soundEnd = document.getElementById("sound-end");

    console.log("Audio elements exist?", {
        item: !!soundItem,
        start: !!soundStart,
        end: !!soundEnd,
    });

    let lastItemsSnapshot = [];
    let lastSessionActive = null;
    let lastTotalAmount = 0;

    // Flag required by browser autoplay rules
    let userInteracted = false;

    // Any click/tap anywhere on the page will unlock audio
    document.addEventListener(
        "pointerdown",
        () => {
            if (!userInteracted) {
                console.log("User interaction detected: sound unlocked.");
            }
            userInteracted = true;
        },
        { once: false }
    );

    function playSound(audioElem, label) {
        if (!audioElem) {
            console.warn(`playSound: audio element for '${label}' is null`);
            return;
        }

        // Respect autoplay rules
        if (!userInteracted) {
            console.warn(
                `playSound('${label}') skipped because user has not interacted with the page yet. Click anywhere once.`
            );
            return;
        }

        try {
            audioElem.currentTime = 0;
            const p = audioElem.play();
            if (p && typeof p.then === "function") {
                p.then(() => {
                    console.log(`Sound '${label}' played successfully.`);
                }).catch((err) => {
                    console.warn(`Sound '${label}' play() blocked or failed:`, err);
                });
            }
        } catch (e) {
            console.warn(`Sound '${label}' exception:`, e);
        }
    }

    function announceTotal(total) {
        if (!("speechSynthesis" in window)) {
            console.warn("Speech synthesis not supported in this browser.");
            return;
        }
        if (!userInteracted) {
            console.warn("Speech synthesis skipped until user interacts with the page.");
            return;
        }

        const num = Number(total) || 0;
        const message = new SpeechSynthesisUtterance(
            `Total amount due is ${num.toFixed(2)} pesos.`
        );
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(message);
        console.log("Speech synthesis: announcing total", num.toFixed(2));
    }

    function hasNewItemOrQuantityIncrease(prevItems, newItems) {
        const prevMap = {};
        for (const it of prevItems) {
            prevMap[it.item_id] = it.quantity;
        }
        for (const it of newItems) {
            const prevQty = prevMap[it.item_id] || 0;
            if (it.quantity > prevQty) {
                return true;
            }
        }
        return false;
    }

    async function fetchPosState() {
        try {
            const response = await fetch("/pos_state", { cache: "no-cache" });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            updatePosUI(data);
        } catch (err) {
            console.error("Failed to fetch POS state:", err);
        }
    }

    function updatePosUI(state) {
        if (!state) return;

        const newSessionActive = !!state.session_active;
        const newItems = state.items || [];
        const newTotal = state.total_amount || 0;
        const lastSessionTotal = state.last_session_total || 0;
        const lastSessionItemCount = state.last_session_item_count || 0;

        const itemAdded = hasNewItemOrQuantityIncrease(
            lastItemsSnapshot,
            newItems
        );

        const prevSessionActive = lastSessionActive;

        if (SESSION_STATUS_ELEMENT) {
            SESSION_STATUS_ELEMENT.textContent = state.session_status || "";

            SESSION_STATUS_ELEMENT.classList.remove("active", "inactive");
            if (newSessionActive) {
                SESSION_STATUS_ELEMENT.classList.add("active");
            } else {
                SESSION_STATUS_ELEMENT.classList.add("inactive");
            }
        }

        if (RECEIPT_BODY_ELEMENT) {
            if (newItems.length === 0) {
                RECEIPT_BODY_ELEMENT.innerHTML =
                    `<tr class="empty-row"><td colspan="4">No items scanned yet</td></tr>`;
            } else {
                const rows = newItems
                    .map((item) => {
                        const priceStr = formatCurrency(item.unit_price);
                        const subtotalStr = formatCurrency(item.subtotal);
                        return `
                            <tr>
                                <td>${escapeHtml(item.name)}</td>
                                <td style="text-align:right;">${item.quantity}</td>
                                <td style="text-align:right;">₱${priceStr}</td>
                                <td style="text-align:right;">₱${subtotalStr}</td>
                            </tr>
                        `;
                    })
                    .join("");
                RECEIPT_BODY_ELEMENT.innerHTML = rows;
            }
        }

        if (TOTAL_VALUE_ELEMENT) {
            TOTAL_VALUE_ELEMENT.textContent = "₱" + formatCurrency(newTotal);
        }

        if (SESSION_SUMMARY_ELEMENT) {
            if (!newSessionActive && lastSessionTotal > 0) {
                SESSION_SUMMARY_ELEMENT.textContent =
                    `Last session total: ₱${formatCurrency(lastSessionTotal)} ` +
                    `(${lastSessionItemCount} item(s))`;
            } else {
                SESSION_SUMMARY_ELEMENT.textContent = "";
            }
        }

        // Session start/end sounds
        if (prevSessionActive !== null && prevSessionActive !== newSessionActive) {
            if (newSessionActive) {
                console.log("Session transitioned: INACTIVE -> ACTIVE (start)");
                playSound(soundStart, "start");
            } else {
                console.log("Session transitioned: ACTIVE -> INACTIVE (end)");
                playSound(soundEnd, "end");
                const announce = lastSessionTotal > 0 ? lastSessionTotal : newTotal;
                if (announce > 0) {
                    announceTotal(announce);
                }
            }
        }

        // Item beep
        if (newSessionActive && itemAdded) {
            console.log("New item or quantity increase detected -> item beep");
            playSound(soundItem, "item");
        }

        lastSessionActive = newSessionActive;
        lastTotalAmount = newTotal;
        lastItemsSnapshot = newItems.map((item) => ({
            item_id: item.item_id,
            quantity: item.quantity,
        }));
    }

    function formatCurrency(value) {
        const num = Number(value) || 0;
        return num.toFixed(2);
    }

    function escapeHtml(str) {
        if (str == null) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    setInterval(fetchPosState, 500);
    fetchPosState();

    console.log("POS frontend script loaded, DOM ready.");
});