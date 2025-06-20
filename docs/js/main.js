// Main JavaScript for RoboDSL website

document.addEventListener('DOMContentLoaded', function() {
    // Toggle submenu functionality
    const submenuToggles = document.querySelectorAll('.has-submenu > a');
    
    submenuToggles.forEach(toggle => {
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            const parent = this.parentElement;
            
            // Close other open submenus
            document.querySelectorAll('.has-submenu.active').forEach(item => {
                if (item !== parent) {
                    item.classList.remove('active');
                }
            });
            
            // Toggle current submenu
            parent.classList.toggle('active');
        });
    });

    // Search functionality
    const searchInput = document.getElementById('search');
    const searchButton = document.querySelector('.search-box button');
    
    function performSearch() {
        const query = searchInput.value.toLowerCase().trim();
        if (query) {
            // Simple search functionality - highlights matching text in content
            highlightSearchResults(query);
        } else {
            clearHighlights();
        }
    }
    
    if (searchInput) {
        searchInput.addEventListener('input', performSearch);
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                performSearch();
            }
        });
    }
    
    if (searchButton) {
        searchButton.addEventListener('click', function(e) {
            e.preventDefault();
            performSearch();
        });
    }
    
    // Simple search highlighting
    function highlightSearchResults(query) {
        clearHighlights();
        
        if (!query) return;
        
        const content = document.getElementById('content');
        if (!content) return;
        
        const walker = document.createTreeWalker(
            content,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        const textNodes = [];
        let node;
        
        while (node = walker.nextNode()) {
            if (node.nodeValue.toLowerCase().includes(query)) {
                textNodes.push(node);
            }
        }
        
        textNodes.forEach(textNode => {
            const parent = textNode.parentNode;
            const text = textNode.nodeValue;
            const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
            const highlightedText = text.replace(regex, '<mark class="search-highlight">$1</mark>');
            
            if (highlightedText !== text) {
                const span = document.createElement('span');
                span.innerHTML = highlightedText;
                parent.replaceChild(span, textNode);
            }
        });
    }
    
    function clearHighlights() {
        const highlights = document.querySelectorAll('.search-highlight');
        highlights.forEach(highlight => {
            const parent = highlight.parentNode;
            parent.replaceChild(document.createTextNode(highlight.textContent), highlight);
            parent.normalize();
        });
    }
    
    function escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add CSS for search highlights
    if (!document.querySelector('#search-highlight-style')) {
        const style = document.createElement('style');
        style.id = 'search-highlight-style';
        style.textContent = `
            .search-highlight {
                background-color: #fbbf24;
                color: #000;
                padding: 0.1em 0.2em;
                border-radius: 2px;
                font-weight: 600;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Mobile menu toggle (if needed for responsive design)
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    const sidebar = document.getElementById('sidebar');
    
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('active');
            document.body.classList.toggle('sidebar-open');
        });
    }
    
    // Close mobile menu when clicking outside
    document.addEventListener('click', function(e) {
        if (window.innerWidth <= 768) {
            const sidebar = document.getElementById('sidebar');
            const sidebarToggle = document.querySelector('.sidebar-toggle');
            
            if (sidebar && sidebar.classList.contains('active') && 
                !sidebar.contains(e.target) && 
                !sidebarToggle?.contains(e.target)) {
                sidebar.classList.remove('active');
                document.body.classList.remove('sidebar-open');
            }
        }
    });
});