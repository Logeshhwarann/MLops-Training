const form = document.querySelector('form');
if (form) {
    form.addEventListener('submit', () => {
        const button = form.querySelector('button[type="submit"]');
        if (button) {
            button.innerText = 'Predicting...';
            button.disabled = true;
        }
    });
}
