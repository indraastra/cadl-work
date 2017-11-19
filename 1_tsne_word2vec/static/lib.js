///
// Borrowed from https://gist.github.com/robertknight/5410420
///
function vecDotProduct(vecA, vecB) {
    var product = 0;
    for (var i = 0; i < vecA.length; i++) {
        product += vecA[i] * vecB[i];
    }
    return product;
}

function vecMagnitude(vec) {
    var sum = 0;
    for (var i = 0; i < vec.length; i++) {
        sum += vec[i] * vec[i];
    }
    return Math.sqrt(sum);
}

function cosineSimilarity(vecA, vecB) {
    return vecDotProduct(vecA, vecB) / (vecMagnitude(vecA) * vecMagnitude(vecB));
}
// End borrowed code.

const DEFAULT_VIZ_OPTS = {
    "height": 600,
    "width": 600,
    "forceAlpha": 0.1,
    "forceAlphaDecay": .005,
    "tsneDim": 2,
    "tsnePerxplexity": 2
}

function lookupEmbedding(words, callback) {
    var wordVecs = [];
    Promise.all(words.map(word => {
      console.log('Fetching: ' + word);
      return $.getJSON('get_embedding/' + word);
    })).then(callback);
}

function visualizeEmbedding(vizElement, words, links, vizOpts = DEFAULT_VIZ_OPTS) {
    $(vizElement).empty();
    const height = vizOpts.height;
    const width = vizOpts.width;

    const svg = d3.select(vizElement).append("svg")
        .attr("width", width)
        .attr("height", height);
    const margin = 40;
    const centerx = d3.scaleLinear()
        .range([width / 2 - height / 2 + margin, width / 2 + height / 2 - margin]);
    const centery = d3.scaleLinear().range([margin, height - margin]);
    const scaleColor = d3.scaleOrdinal(d3.schemeCategory20);

    const tsne = new tsnejs.tSNE({
        dim: 2,
        perplexity: 10,
    });
    // Compute pairwise distances to initialize the t-SNE model..
    const dists = words.map(d => words.map(e => 1 - cosineSimilarity(d.vec, e.vec)));
    tsne.initDataDist(dists);

    const forcetsne = d3.forceSimulation(
      words.map(d => (d.x = width / 2, d.y = height / 2, d)))
        .alphaDecay(vizOpts.forceAlphaDecay)
        .alpha(vizOpts.forceAlpha)
        .force('tsne', function (alpha) {
            // Get the solution for this timestep.
            tsne.step();
            const points = tsne.getSolution();

            // Set the domains to the x and y limits of the solution so that
            // the plot is always within the bounds.
            centerx.domain(d3.extent(points.map(d => d[0])));
            centery.domain(d3.extent(points.map(d => d[1])));

            // Update the position of each word.
            words.forEach((d, i) => {
                d.x += alpha * (centerx(points[i][0]) - d.x);
                d.y += alpha * (centery(points[i][1]) - d.y);
                d.color = scaleColor(i);
            });
        })
        .on('tick', function () {
            draw(svg, words);
        });

    function draw(svg, words) {
        var svg = d3.select('svg');
        var link = svg
            .selectAll(".link")
            .data(links)
        link.enter()
            .append("line")
            .attr("class", "link")
            .merge(link)
            .attr("x1", function(d) {
                return d.source.x;
            })
            .attr("y1", function(d) {
                return d.source.y;
            })
            .attr("x2", function(d) {
                return d.target.x;
            })
            .attr("y2", function(d) {
                return d.target.y;
            })
            .attr("stroke", "gray")
            .attr("stroke-width", 2);

        var node = svg
            .selectAll('.node')
            .data(words);

        enter = node.enter().append('g')
            .attr('class', 'node');
        enter.append('circle')
            .attr('r', 35)
            .attr('stroke', function(d) {
                return d.color;
            })
            .attr("stroke-width", 10)
            .attr("fill-opacity", 0);
        enter.append('text')
              .text(function(d) { return d.word; });

        enter.merge(node)
              .attr('transform', function(d) {
                return "translate(" + d.x + "," + d.y + ")";
              });

        node.exit().remove()
    }
}
