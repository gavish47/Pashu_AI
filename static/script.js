function uploadImage() {

    let file = document.getElementById("fileInput").files[0];

    if (!file) {
        alert("Upload image first");
        return;
    }

    document.getElementById("previewImage").src =
        URL.createObjectURL(file);

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {

        document.getElementById("breed").innerText = data.breed;
        document.getElementById("confidence").innerText = data.confidence + "%";
        document.getElementById("origin").innerText = data.origin;
        document.getElementById("description").innerText = data.description;

        document.getElementById("others").innerHTML =
            data.others.map(o => `• ${o.breed} (${o.conf}%)`).join("<br>");
    });
}