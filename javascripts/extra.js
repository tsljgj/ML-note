document.addEventListener("DOMContentLoaded", function() {
    var elements = document.getElementsByTagName('p'); // Targets all paragraph elements
    for (var i = 0; i < elements.length; i++) {
        elements[i].childNodes.forEach(function(node) {
            if (node.nodeType === 3) { // Node type 3 is a text node
                node.nodeValue = node.nodeValue
                    .replace(/"([^"]+)"/g, '“$1”') // Replaces "quote" with “quote”
                    .replace(/(^|\s)'(\w)/g, '$1‘$2') // Converts leading single quote to open curly quote
                    .replace(/(\w)'(\s|$|,|\.|\?|!)/g, '$1’$2'); // Converts trailing single quote to close curly quote
            }
        });
    }
});
