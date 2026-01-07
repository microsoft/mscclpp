// Version selector for sphinx-multiversion documentation
(function() {
    'use strict';
    
    // Configuration: list all available versions
    // This should be updated when new versions are released
    const versions = [
        { name: 'latest (v0.8.0)', path: '', version: 'latest' },
        { name: 'v0.8.0', path: 'v0.8.0', version: 'v0.8.0' },
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
        const match = path.match(/\/(v\d+\.\d+\.\d+)\//);
        return match ? match[1] : 'latest';
    }
    
    function getBasePath() {
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
        
        // Create version selector container
        const selectorDiv = document.createElement('div');
        selectorDiv.style.padding = '10px';
        selectorDiv.style.borderBottom = '1px solid #404040';
        selectorDiv.style.marginTop = '10px';
        
        const label = document.createElement('label');
        label.htmlFor = 'version-selector';
        label.style.color = '#b3b3b3';
        label.style.fontSize = '90%';
        label.textContent = 'Version:';
        
        const select = document.createElement('select');
        select.id = 'version-selector';
        select.style.width = '100%';
        select.style.marginTop = '5px';
        select.style.padding = '5px';
        select.style.backgroundColor = '#2c2c2c';
        select.style.color = '#ffffff';
        select.style.border = '1px solid #404040';
        select.style.borderRadius = '3px';
        
        // Add options
        versions.forEach(function(version) {
            const option = document.createElement('option');
            const isSelected = currentVersion === version.version;
            
            // Build the URL relative to current page
            let url = basePath + (version.path ? version.path + '/' : '');
            
            // Try to preserve the current page
            const currentPath = window.location.pathname;
            const pageName = currentPath.split('/').pop();
            if (pageName && pageName !== '' && pageName.endsWith('.html')) {
                url += pageName;
            } else {
                url += 'index.html';
            }
            
            option.value = url;
            option.textContent = version.name;
            if (isSelected) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        
        select.addEventListener('change', function() {
            if (this.value) {
                window.location.href = this.value;
            }
        });
        
        selectorDiv.appendChild(label);
        selectorDiv.appendChild(select);
        
        // Insert after the search form
        searchDiv.appendChild(selectorDiv);
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createVersionSelector);
    } else {
        createVersionSelector();
    }
})();
