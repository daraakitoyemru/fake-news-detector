document.querySelector("#checkText").addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        function: getSelectedText,
      },
      (results) => {
        if (results && results[0] && results[0].result) {
          // Handle the selected text here (e.g., send it to your backend API)
          document.querySelector("#result").textContent = results[0].result;
        } else {
          document.querySelector("result").textContent = "No text selected!";
        }
      }
    );
  });
});

// Function to run in the content script to get highlighted text
function getSelectedText() {
  return window.getSelection().toString();
}
