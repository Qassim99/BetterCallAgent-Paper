# Security policy

## Reporting

Report a suspected vulnerability through the private contact channel supplied by the
paper authors. Do not include credentials, legal documents, or personally identifying
query content in a public issue. A permanent public security contact has not yet been
published and is a release requirement.

## Development and deployment

Documented development servers bind to `127.0.0.1`. A network deployment must add:

1. TLS termination;
2. authentication and authorization;
3. restricted CORS origins;
4. rate and request-size limits;
5. provider spending and timeout controls; and
6. logs that exclude credentials and unnecessary document text.

The backend allowlists model identifiers. Provider credentials remain server-side,
TLS verification is mandatory, and live provider failures do not fall back to
fixture output. Every `VITE_*` variable is public in the browser bundle.

## Secrets and publication

Copy only `.env.example`; never commit `.env`, `.env.local`, provider credentials,
Kaggle credentials, SSH keys, token files, or environment snapshots. If a credential
enters Git history, revoke it immediately and remove the affected objects before
publication.

The audited predecessor development history may have contained potentially live
credentials. All affected credentials must be rotated, and that history must remain
outside the clean release repository.

The clean source repository may be publicly visible, but it must not be presented as
an open-source or complete artifact release until:

- credential rotation is complete;
- the source history passes `python scripts/check_release.py`;
- a software license and contributor terms are approved; and
- data and derived-artifact redistribution rights are confirmed.

## Supported versions

Security fixes target the latest release on the default branch. Immutable research
snapshots may not receive patches; do not deploy a snapshot merely because it is
citable.
