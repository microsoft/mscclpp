// Version selector for sphinx-multiversion documentation
(function() {
    'use strict';
    
    // Configuration: list all available versions
    // This should be updated when new versions are released
    const versions = [
        { name: 'main (dev)', path: '', version: 'main' },
        { name: 'v0.8.0 (latest)', path: 'v0.8.0', version: 'v0.8.0' },
        { name: 'v0.7.0', path: 'v0.7.0', version: 'v0.7.0' },
        { name: 'v0.6.0', path: 'v0.6.0', version: 'v0.6.0' },
        { name: 'v0.5.2', path: 'v0.5.2', version: 'v0.5.2' },
        { name: 'v0.5.1', path: 'v0.5.1', version: 'v0.5.1' },
        { name: 'v0.5.0', path: 'v0.5.0', version: 'v0.5.0' },
        { name: 'v0.4.3', path: 'v0.4.3', version: 'v0.4.3' },
        { name: 'v0.4.2', path: 'v0.4.2', version: 'v0.4.2' },
        { name: 'v0.4.1', path: 'v0.4.1', version: 'v0.4.1' },
        { name: 'v0.4.0', path: 'v0.4.0', version: 'v0.4.0' }
    ];
    
    function detectCurrentVersion() {
        const path = window.location.pathname;
        // Check for version tags first
        const match = path.match(/\/(v\d+\.\d+\.\d+)\//);
        if (match) {
            return match[1];
        }
        // Check for main branch directory
        if (path.includes('/main/')) {
            return 'main';
        }
        // If at root (no version in path), it's main
        return 'main';
    }    function getBasePath() {
        const path = window.location.pathname;
        // Find how many levels deep we are from the version directory
        const match = path.match(/\/(v\d+\.\d+\.\d+|)\/(.*)/);
        if (match) {
            const depth = match[2].split('/').filter(p => p && p !== 'index.html').length;
            return '../'.repeat(depth + 1);
        }
        // For root level (latest), calculate depth
        const segments = path.split('/').filter(p => p && p !== 'index.html');
        return '../'.repeat(segments.length);
    }
    
    function createVersionSelector() {
        const currentVersion = detectCurrentVersion();
        const basePath = getBasePath();
        const searchDiv = document.querySelector('.wy-side-nav-search');
        
        if (!searchDiv) return;

        // Find the title link (mscclpp)
        const titleLink = searchDiv.querySelector('a.icon-home');
        
        // Create version selector container
        const selectorDiv = document.createElement('div');
        selectorDiv.style.padding = '10px';
        selectorDiv.style.paddingTop = '5px';
        selectorDiv.style.paddingBottom = '10px';
        
        const select = document.createElement('select');
        select.id = 'version-selector';
        select.style.width = '100%';
        select.style.padding = '5px';
        select.style.backgroundColor = '#2c2c2c';
        select.style.color = '#ffffff';
        select.style.border = '1px solid #404040';
        select.style.borderRadius = '3px';
        
        // Add options
        versions.forEach(function(version) {
            const option = document.createElement('option');
            const isSelected = currentVersion === version.version;
            
            // Build the URL - use absolute paths from root
            let url;
            const currentPath = window.location.pathname;
            const pageName = currentPath.split('/').pop() || 'index.html';

            if (version.version === 'main' && version.path === '') {
                // For main (dev) at root
                url = '/' + pageName;
            } else {
                // For versioned releases
                url = '/' + version.path + '/' + pageName;
            }
            
            console.log('[Version Selector] Option: ' + version.name + ', absoluteURL=' + url);

            option.value = url;
            console.log('[Version Selector] After setting option.value, option.value=' + option.value);
            option.textContent = version.name;
            if (isSelected) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        
        select.addEventListener('change', function() {
            console.log('[Version Selector] Change event triggered');
            console.log('[Version Selector] Selected value:', this.value);
            console.log('[Version Selector] Current location:', window.location.href);
            if (this.value) {
                console.log('[Version Selector] Navigating to:', this.value);
                window.location.href = this.value;
            }
        });
        
        selectorDiv.appendChild(select);
        
        // Insert after the title link in the searchDiv
        if (titleLink) {
            // Insert after the title link element
            const nextElement = titleLink.nextSibling;
            if (nextElement) {
                searchDiv.insertBefore(selectorDiv, nextElement);
            } else {
                searchDiv.appendChild(selectorDiv);
            }
        } else {
            // Fallback: insert at the beginning of searchDiv
            searchDiv.insertBefore(selectorDiv, searchDiv.firstChild);
        }
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createVersionSelector);
    } else {
        createVersionSelector();
    }
})();
