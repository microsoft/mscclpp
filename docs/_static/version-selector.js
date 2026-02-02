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

    /**
     * Detect the base path for GitHub Pages project sites.
     * For project sites, the URL is like /repository-name/v0.8.0/guide.html
     * For root sites or local development, the URL is like /v0.8.0/guide.html
     * @returns {string} The base path (e.g., '/mscclpp' or '')
     */
    function detectBasePath() {
        // Most reliable method: detect from this script's own URL
        // The script is always at {base}/_static/version-selector.js or {base}/vX.Y.Z/_static/version-selector.js
        const scripts = document.getElementsByTagName('script');
        for (let i = 0; i < scripts.length; i++) {
            const src = scripts[i].src;
            if (src && (src.includes('/_static/version-selector.js') || src.endsWith('version-selector.js'))) {
                try {
                    const url = new URL(src);
                    const scriptPath = url.pathname;
                    // Extract base path: everything before /_static/version-selector.js
                    // But also strip version directories like /v0.8.0/ or /main/
                    const match = scriptPath.match(/^(.*?)\/_static\/version-selector\.js$/);
                    if (match) {
                        let basePath = match[1] || '';
                        // Remove version suffix if present (e.g., /mscclpp/v0.8.0 -> /mscclpp)
                        basePath = basePath.replace(/\/(v\d+\.\d+\.\d+|main)$/, '');
                        return basePath;
                    }
                } catch (e) {
                    // URL parsing failed, continue to fallback
                    // Log a warning to aid debugging when the primary detection method fails.
                    if (typeof console !== 'undefined' && typeof console.warn === 'function') {
                        console.warn('version-selector: Failed to parse script URL for base path detection; falling back to location-based detection.', e);
                    }
                }
            }
        }

        // Fallback: try to detect from URL path
        const path = window.location.pathname;
        const segments = path.split('/').filter(s => s.length > 0);

        if (segments.length >= 1) {
            const firstSegment = segments[0];
            // If first segment is not a version tag (vX.Y.Z), not 'main', and
            // does not look like a file name (no '.' in the segment), then it's
            // the GitHub Pages project base path (e.g., 'mscclpp').
            // This handles both:
            //   /mscclpp/v0.8.0/index.html -> base is /mscclpp
            //   /mscclpp/index.html -> base is /mscclpp
            // while avoiding treating root files like /index.html as a base path.
            if (!firstSegment.match(/^v\d+\.\d+\.\d+$/) && firstSegment !== 'main' && !firstSegment.includes('.')) {
                return '/' + firstSegment;
            }
        }

        // No base path (root site or local development)
        return '';
    }
    
    function detectCurrentVersion() {
        const path = window.location.pathname;
        // Check for version tags first
        // Match version tags in the format v0.0.0 within the URL path
        // This works for both /v0.8.0/... and /mscclpp/v0.8.0/...
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
        const basePath = detectBasePath();
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
            
            // Build the URL - use absolute paths with base path for GitHub Pages
            let url;
            const currentPath = window.location.pathname;

            // Extract the page path relative to the version directory
            // For /mscclpp/v0.7.0/design/design.html -> design/design.html
            // For /v0.7.0/design/design.html -> design/design.html
            // For /mscclpp/index.html -> index.html
            // For /index.html -> index.html
            let relativePath;
            
            // Remove base path first if present
            let pathWithoutBase = currentPath;
            if (basePath && currentPath.startsWith(basePath)) {
                pathWithoutBase = currentPath.substring(basePath.length);
            }
            
            const versionMatch = pathWithoutBase.match(/^\/(v\d+\.\d+\.\d+)\/(.*)/);
            if (versionMatch) {
                // We're in a versioned directory
                relativePath = versionMatch[2] || 'index.html';
            } else {
                // We're at root (main/dev) - remove leading slash
                relativePath = pathWithoutBase.substring(1) || 'index.html';
                // Handle /main/ path
                const mainMatch = pathWithoutBase.match(/^\/main\/(.*)/);
                if (mainMatch) {
                    relativePath = mainMatch[1] || 'index.html';
                }
            }

            if (version.version === 'main' && version.path === '') {
                // For main (dev) at root
                url = basePath + '/' + relativePath;
            } else {
                // For versioned releases
                url = basePath + '/' + version.path + '/' + relativePath;
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
                const versionMatch = baseUrl.match(/\/(v\d+\.\d+\.\d+)\//);
                if (versionMatch) {
                    fallbackUrl = basePath + '/' + versionMatch[1] + '/index.html';
                } else {
                    fallbackUrl = basePath + '/index.html';
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
