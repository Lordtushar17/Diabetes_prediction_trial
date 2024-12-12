// Add a subtle bounce animation when the card is hovered
document.addEventListener("DOMContentLoaded", () => {
    const card = document.querySelector(".card");
    card.addEventListener("mouseover", () => {
        card.style.transform = "scale(1.05)";
    });
    card.addEventListener("mouseleave", () => {
        card.style.transform = "scale(1)";
    });
});
