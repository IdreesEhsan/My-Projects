Dropzone.autoDiscover = false;

function init() {

    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        autoProcessQueue: false
    });

    dz.on("addedfile", function () {
        if (dz.files.length > 1) {
            dz.removeFile(dz.files[0]);
        }
    });

    dz.on("complete", function (file) {

        let url = "http://127.0.0.1:5000/classify_image";

        $.post(url, {
            image_data: file.dataURL
        }, function (data, status) {

            console.log(data);

            if (!data || data.length === 0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }

            /* ✅ UPDATED: Added Ronaldo & Verstappen */
            let players = [
                "lionel_messi",
                "maria_sharapova",
                "roger_federer",
                "serena_williams",
                "virat_kohli",
                "cristiano_ronaldo",
                "max_verstappen"
            ];

            let match = null;
            let bestScore = -1;

            for (let i = 0; i < data.length; ++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if (maxScoreForThisClass > bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }

            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();

                /* Show matched player card */
                $("#resultHolder").html(
                    $(`[data-player="${match.class}"]`).html()
                );

                /* Fill probability table */
                let classDictionary = match.class_dictionary;

                for (let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let probabilityScore = match.class_probability[index];
                    $("#score_" + personName).html(probabilityScore.toFixed(2));
                }
            }
        });
    });

    $("#submitBtn").on("click", function () {
        dz.processQueue();
    });
}

$(document).ready(function () {
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    init();
});
