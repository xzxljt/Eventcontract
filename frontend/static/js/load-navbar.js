// load-navbar.js

/**
 * Loads the navbar HTML from a file and inserts it into a specified container.
 * Highlights the active link based on the current page path.
 * @param {string} containerId - The ID of the HTML element where the navbar should be inserted.
 * @param {string} navbarPath - The path to the navbar HTML file.
 */
async function loadNavbar(containerId, navbarPath) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Navbar container with ID "${containerId}" not found.`);
        return;
    }

    try {
        const response = await fetch(navbarPath);
        if (!response.ok) {
            throw new Error(`Failed to load navbar: ${response.statusText}`);
        }
        const navbarHtml = await response.text();
        container.innerHTML = navbarHtml;

        // Highlight the active link based on the current page path
        const currentPath = window.location.pathname;
        const navLinks = container.querySelectorAll('.nav-link');

        navLinks.forEach(link => {
            // Remove existing active class
            link.classList.remove('active');
            link.removeAttribute('aria-current');

            // Determine the href to match (handle root path '/')
            const linkHref = link.getAttribute('href');
            const normalizedLinkHref = linkHref === '/' ? '/' : linkHref.replace(/\/$/, ''); // Remove trailing slash unless it's just '/'
            const normalizedCurrentPath = currentPath === '/' ? '/' : currentPath.replace(/\/$/, ''); // Remove trailing slash unless it's just '/'

            // Check if the link's href matches the current path
            if (normalizedLinkHref === normalizedCurrentPath) {
                link.classList.add('active');
                link.setAttribute('aria-current', 'page');
            }
        });

    } catch (error) {
        console.error("Error loading or processing navbar:", error);
        container.innerHTML = "<p class='text-danger'>Failed to load navigation bar.</p>"; // Display error message
    }
}

// Example usage (will be called from each HTML file)
// document.addEventListener('DOMContentLoaded', () => {
//     loadNavbar('navbar-container', '/templates/navbar.html');
// });