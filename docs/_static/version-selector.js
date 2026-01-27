// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * Version selector for sphinx-multiversion documentation.
 *
 * The DEFINED_VERSIONS array is auto-generated from git tags by generate_versions.py
 * which runs automatically during 'make html'. This ensures the version list stays
 * in sync with sphinx-multiversion without manual updates.
 *
 * The versions.js file (loaded before this script) defines DEFINED_VERSIONS.
 */
(function() {
    'use strict';

    // DEFINED_VERSIONS is defined in versions.js (auto-generated from git tags)
    // Fallback to main only if versions.js failed to load
    const versions = (typeof DEFINED_VERSIONS !== 'undefined') ? DEFINED_VERSIONS : [
        { name: 'main (dev)', path: '', version: 'main' }
    ];
    
    function detectCurrentVersion() {
        const path = window.location.pathname;
        // Check for version tags first
        // Match version tags in the format v0.0.0 within the URL path
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
    
    function createVersionSelector() {
        const currentVersion = detectCurrentVersion();
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

            option.value = url;
            option.textContent = version.name;
            if (isSelected) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        
        select.addEventListener('change', function() {
            if (this.value) {
                const baseUrl = this.value;
                const currentHash = window.location.hash;
                const targetUrl = baseUrl + currentHash;

                // Calculate fallback URL (version index page)
                let fallbackUrl;
                const versionMatch = baseUrl.match(/^\/(v\d+\.\d+\.\d+)\//);
                if (versionMatch) {
                    fallbackUrl = '/' + versionMatch[1] + '/index.html';
                } else {
                    fallbackUrl = '/index.html';
                }

                // Check if the target page exists using HEAD request
                const controller = new AbortController();
                const timeoutId = setTimeout(function() { controller.abort(); }, 2000);

                fetch(baseUrl, {
                    method: 'HEAD',
                    signal: controller.signal
                })
                    .then(function(response) {
                        clearTimeout(timeoutId);
                        if (response.ok) {
                            window.location.href = targetUrl;
                        } else {
                            // Page doesn't exist, navigate to version index
                            window.location.href = fallbackUrl;
                        }
                    })
                    .catch(function(error) {
                        clearTimeout(timeoutId);
                        // On network error or timeout, try fallback first
                        // This handles cases where the page truly doesn't exist
                        window.location.href = fallbackUrl;
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
