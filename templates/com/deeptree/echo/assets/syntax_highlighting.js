function highlightCode(language, codeBase64) {
  const code = fromBase64(codeBase64);

  const knownLanguage = !!hljs.getLanguage(language);
  const value = knownLanguage
    ? hljs.highlight(code, { language })
    : hljs.highlightAuto(code);

  const container = document.createElement("code", {});
  container.classList.add("hljs");

  container.innerHTML = value.value;
  document.body.replaceWith(container);

  function nodeToObj(node) {
    let style = {};
    if (node.nodeName !== "#text") {
      style = window.getComputedStyle(node);
    }
    const obj = {};
    obj.value = node.textContent;
    obj.color = rgbToArr(style.color);
    // 400 is default.
    if (style.fontWeight && style.fontWeight !== "400") {
      obj.fontWeight = parseInt(style.fontWeight);
    }
    obj.children = [];
    if (node.childNodes.length > 0) {
      for (let i = 0; i < node.childNodes.length; i++) {
        obj.children.push(nodeToObj(node.childNodes[i]));
      }
    }
    return obj;
  }

  const containerStyle = window.getComputedStyle(container);

  return {
    background: rgbToArr(containerStyle.background),
    color: rgbToArr(containerStyle.color),
    children: [...container.childNodes].map(nodeToObj),
  };
}

function rgbToArr(rgb) {
  if (!rgb) return null;
  return rgb
    .replace("rgb(", "")
    .replace("rgba(", "")
    .replace(")", "")
    .split(", ")
    .map((v) => parseFloat(v));
}

/**
 * This improves on atob by adding support for unicode characters.
 * https://stackoverflow.com/a/30106551
 */
function fromBase64(str) {
  return decodeURIComponent(
    atob(str)
      .split("")
      .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
      .join("")
  );
}
