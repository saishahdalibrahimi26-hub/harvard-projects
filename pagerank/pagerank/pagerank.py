import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probabilities = {}

    pages = list(corpus.keys())
    n = len(pages)
    links = corpus[page]

    if len(links) == 0:
         links = set(pages)

    for p in pages:
        probabilities[p] = (1 - damping_factor) / n

    for link in links:
        probabilities[link] += damping_factor / len(links)

    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())

    counts = {}

    for page in pages:
        counts[page] = 0

    current_page = random.choice(pages)
    counts[current_page] += 1

    for _ in range(n - 1):
        model = transition_model(corpus, current_page, damping_factor)

        current_page = random.choices(
            list(model.keys()),
            weights=list(model.values())
        )[0]

        counts[current_page] += 1

    return {
        page: counts[page] / n
        for page in pages
    }




def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    n = len(pages)

    ranks = {}

    for page in pages:
        ranks[page] = 1 / n

    while True:
        new_ranks = {}

        for page in pages:
            total = 0

            for other_page in pages:
                links = corpus[other_page]

                if len(links) == 0:
                    links = set(pages)
                if page in links:
                    total += ranks[other_page] / len(links)

            new_ranks[page] = (
                (1 - damping_factor) / n
                + damping_factor * total
            )
        difference = max(
            abs(new_ranks[page] - ranks[page])
            for page in pages
        )

        ranks = new_ranks

        if difference < 0.001:
            break
    return ranks

        



if __name__ == "__main__":
    main()
