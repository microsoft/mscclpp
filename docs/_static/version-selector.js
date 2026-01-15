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
    }

    function getBasePath() {
        const path = window.location.pathname;
        // Find how many levels deep we are from the version directory
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
            
            // Build the URL - use absolute paths from root (without hash)
            let url;
            const currentPath = window.location.pathname;

            // Extract the page path relative to the version directory
            // For /v0.7.0/design/design.html -> design/design.html
            // For /index.html -> index.html
            let relativePath;
            const versionMatch = currentPath.match(/^\/(v\d+\.\d+\.\d+)\/(.*)/);
            if (versionMatch) {
                // We're in a versioned directory
                relativePath = versionMatch[2] || 'index.html';
            } else {
                // We're at root (main/dev)
                relativePath = currentPath.substring(1) || 'index.html';
            }

            if (version.version === 'main' && version.path === '') {
                // For main (dev) at root
                url = '/' + relativePath;
            } else {
                // For versioned releases
                url = '/' + version.path + '/' + relativePath;
            }
            
            console.log('[Version Selector] Option: ' + version.name + ', relativePath=' + relativePath + ', absoluteURL=' + url);

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
                const baseUrl = this.value;
                const currentHash = window.location.hash; // Get current hash at selection time
                const targetUrl = baseUrl + currentHash; // Append current hash to target URL

                console.log('[Version Selector] Current hash:', currentHash);
                console.log('[Version Selector] Target URL with hash:', targetUrl);
                console.log('[Version Selector] Checking if page exists:', baseUrl);

                // Check if the target page exists using a fetch with abort
                const controller = new AbortController();
                const timeoutId = setTimeout(function() { controller.abort(); }, 1000);

                fetch(baseUrl, {
                    method: 'GET',
                    signal: controller.signal
                })
                    .then(function(response) {
                        clearTimeout(timeoutId);
                        console.log('[Version Selector] Response status:', response.status);
                        if (response.ok) {
                            // Page exists, navigate to it with hash
                            console.log('[Version Selector] Page exists, navigating to:', targetUrl);
                            window.location.href = targetUrl;
                        } else {
                            // Page doesn't exist, fall back to version root index.html
                            // For versioned paths like /v0.8.0/... -> /v0.8.0/index.html
                            // For root paths like /py_api/... -> /index.html
                            let fallbackUrl;
                            const versionMatch = baseUrl.match(/^\/(v\d+\.\d+\.\d+)\//);
                            if (versionMatch) {
                                // It's a versioned path
                                fallbackUrl = '/' + versionMatch[1] + '/index.html';
                            } else {
                                // It's a root path (main/dev)
                                fallbackUrl = '/index.html';
                            }
                            console.log('[Version Selector] Page not found (status: ' + response.status + '), falling back to:', fallbackUrl);
                            window.location.href = fallbackUrl;
                        }
                    })
                    .catch(function(error) {
                        clearTimeout(timeoutId);
                        // On error (including timeout), try to navigate anyway
                        console.log('[Version Selector] Error checking page:', error.name, error.message);
                        console.log('[Version Selector] Navigating to target anyway:', targetUrl);
                        window.location.href = targetUrl;
                    });
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
