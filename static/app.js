function readFile(input) {
    if (input.files && input.files[0]) {
        let reader = new FileReader();

        reader.onload = function (e) {
            let htmlPreview =
                '<img width="200" src="' +
                e.target.result +
                '"/>' +
                "<p>" +
                input.files[0].name +
                "</p>";
            let wrapperZone = $(input).parent();
            let previewZone = $(input)
                .parent()
                .parent()
                .find(".preview-zone");
            let boxZone = $(input)
                .parent()
                .parent()
                .find(".preview-zone")
                .find(".box")
                .find(".box-body");

            wrapperZone.removeClass("dragover");
            previewZone.removeClass("hidden");
            boxZone.empty();
            boxZone.append(htmlPreview);
        };

        reader.readAsDataURL(input.files[0]);
    }
}

function reset(e) {
    e.wrap("<form>")
        .closest("form")
        .get(0)
        .reset();
    e.unwrap();
}

$(".dropzone").change(function () {
    readFile(this);
});

$(".dropzone-wrapper").on("dragover", function (e) {
    e.preventDefault();
    e.stopPropagation();
    $(this).addClass("dragover");
});

$(".dropzone-wrapper").on("dragleave", function (e) {
    e.preventDefault();
    e.stopPropagation();
    $(this).removeClass("dragover");
});

$(".remove-preview").on("click", function () {
    let boxZone = $(this)
        .parents(".preview-zone")
        .find(".box-body");
    let previewZone = $(this).parents(".preview-zone");
    let dropzone = $(this)
        .parents(".form-group")
        .find(".dropzone");
    boxZone.empty();
    previewZone.addClass("hidden");
    reset(dropzone);
});
