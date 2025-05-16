# Setting Up GitHub Pages for Your MLOps Demo

Follow these steps to create a live demo of your MLOps project on GitHub Pages:

## 1. Configure GitHub Pages in Your Repository

1. Go to your repository on GitHub
2. Navigate to **Settings** > **Pages**
3. Under **Source**, select `main` branch
4. Click **Save**

## 2. Ensure Demo Files Are in the Repository Root

The following files should be in your repository root:
- `index.html` - Main demo page
- `kubeflow_to_vertex.html` - Migration guide
- `pipeline_visualization.png` - Pipeline diagram
- `evaluation_metrics.png` - Model metrics visualization
- `monitoring_dashboard.png` - Monitoring dashboard
- `positive_explanation.png` - Model explanation visualization

## 3. Update Your Repository README

Add the GitHub Pages URL to your README:

```markdown
## Live Demo

Check out our interactive demo: https://kiriti-v.github.io/mlops-demo/
```

## 4. Customize for Your Project

1. Update the GitHub repository link in `index.html`:
   ```html
   <a href="https://github.com/YOUR-USERNAME/mlops-demo" class="button">View on GitHub</a>
   ```

2. Update any project-specific information in the HTML files

## 5. Check Your Live Demo

Once GitHub Pages is set up, your demo will be available at:
`https://kiriti-v.github.io/mlops-demo/`

It may take a few minutes for changes to appear after pushing to your repository.

## 6. Add More Content (Optional)

Consider adding these additional pages:
- Architecture diagram with GCP services
- Detailed component documentation
- Interactive API demo
- Interactive pipeline visualization

## 7. Share Your Demo

When applying for roles, include the GitHub Pages URL in your:
- Resume
- Cover letter
- LinkedIn profile
- Portfolio 