
Collect aggregated content for each type in `agglom/{typename}` such as assets, org, res, etc.

- Deduplicate identical files.
- For unique files like manifests, add `-{manifestname}.xml` to the end of each one to prepare for later consolidation.

- Collect aggregated content for each type in `agglom/{typename}`, such as assets, org, res, etc., ensuring that identical files are deduplicated. Where certain files like manifests are unique, append `-{manifestname}.xml` to the end of each one so they can be consolidated later.

- Investigate aggregates further and identify key differentiators for each type.
  - Document findings in `docs/{typename}_analysis.md` to track variations across different `agglom` types.
  - Utilize patterns and unique naming conventions to ensure consistency and comprehension.
- 