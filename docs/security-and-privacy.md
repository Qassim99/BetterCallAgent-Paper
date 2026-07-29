# Security and privacy

The browser communicates only with the BetterCallAgent backend. Provider credentials
remain server-side.

A real online query can send the question and selected document excerpts to the
configured LLM provider. Before using confidential or personal legal material, verify
the provider's location, retention policy, data-processing agreement, and applicable
law.

Development commands bind to loopback. For network deployment:

1. terminate TLS at a reverse proxy;
2. require authentication;
3. restrict allowed origins;
4. rate-limit requests and cap body sizes;
5. set provider spending limits;
6. use separate chat and embedding credentials; and
7. keep audit logs free of secrets and unnecessary document text.

TLS verification is enabled. The release does not offer a configuration switch that
silently disables certificate validation.

Predecessor development history may have contained live credentials. All potentially
affected credentials must be rotated before publication, and predecessor Git objects
must not be included in the clean release history.

Public release also requires an approved software license, contributor agreement, and
confirmation that every distributed data or derived artifact may be redistributed.
