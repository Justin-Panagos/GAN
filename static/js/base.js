document.getElementById('generateForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Show loading indicator
    document.getElementById('loading').style.display = 'block';

    var textInput = document.getElementById('textInput').value;
    var imageInput = document.getElementById('imageInput').files[0];

    if (!textInput && !imageInput) {
        alert("Please provide either text or an image.");
        document.getElementById('loading').style.display = 'none';
        return;
    }

    var formData = new FormData();
    if (imageInput) formData.append('image', imageInput);
    if (textInput) formData.append('text', textInput);

    // Send AJAX request to generate image
    fetch('/generate/', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            document.getElementById('loading').style.display = 'none';

            if (data.error) {
                alert("Error: " + data.error);
            } else {
                // Create an image element and display the result
                var img = document.createElement("img");
                img.src = "data:image/png;base64," + data.image;
                document.getElementById('result').appendChild(img);
            }
        })
        .catch(error => {
            console.error("Error:", error);
            document.getElementById('loading').style.display = 'none';
        });
});
