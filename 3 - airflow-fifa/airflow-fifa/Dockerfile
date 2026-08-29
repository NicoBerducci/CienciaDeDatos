FROM astrocrpublic.azurecr.io/runtime:3.3-4

# El código de la cátedra vive en include/fifa y se importa como `fifa`.
ENV PYTHONPATH="/usr/local/airflow/include:${PYTHONPATH}"

# Chromium para la ruta de respaldo (ver el docstring del DAG).
# Es lo que hace lento el primer `astro dev start`: se paga una vez y queda
# cacheado. Si no lo necesitás, comentá este bloque y usá engine=http.
USER root
RUN PLAYWRIGHT_BROWSERS_PATH=/usr/local/share/ms-playwright \
    playwright install --with-deps chromium \
    && chmod -R a+rX /usr/local/share/ms-playwright
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/share/ms-playwright
USER astro
