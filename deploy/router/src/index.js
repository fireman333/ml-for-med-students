// med-study-rpg-ml-router — reverse-proxy Worker for the ML intro site under
// https://med-study-rpg.com/ml. Modeled on klaudehealthedu-router.
//
// The Pages project is uploaded with the site nested under /ml/ (dist-deploy/ml/…),
// so the pathname is forwarded UNCHANGED and MkDocs' relative links resolve.

const ORIGIN = "https://med-study-rpg-ml.pages.dev";
const PREFIX = "/ml";

export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Bare /ml -> /ml/ so relative asset paths resolve against the directory.
    if (url.pathname === PREFIX) {
      return Response.redirect(url.origin + PREFIX + "/" + url.search, 301);
    }
    if (!url.pathname.startsWith(PREFIX + "/")) {
      return new Response("Not found", { status: 404 });
    }

    const upstream = await fetch(ORIGIN + url.pathname + url.search, request);
    const resp = new Response(upstream.body, upstream);
    resp.headers.delete("content-encoding");
    resp.headers.delete("content-length");
    resp.headers.set("x-served-by", "edge-router-ml");
    return resp;
  },
};
