# Deploy to https://med-study-rpg.com/ml/

Prereq: `wrangler login` (token must be valid: `wrangler whoami`).

```bash
# 1. one-time: create the Pages project
wrangler pages project create med-study-rpg-ml --production-branch main
# 2. build (nested under /ml/)
bash deploy/build.sh
# 3. upload to Pages
wrangler pages deploy dist-deploy --project-name med-study-rpg-ml --branch main
# 4. verify pages.dev origin before binding routes
curl -sI https://med-study-rpg-ml.pages.dev/ml/ | head -3
# 5. deploy router Worker (binds med-study-rpg.com/ml routes)
cd deploy/router && wrangler deploy
# 6. verify it is really served by the router, not the root SPA
curl -sI https://med-study-rpg.com/ml/ | grep -i x-served-by   # expect edge-router-ml
```

Hub button: repo fireman333/study-rpg, `scripts/cf-landing-template.html` 網站 section
(see research/05_site_style_deploy.md). Push to main triggers public deploy.
