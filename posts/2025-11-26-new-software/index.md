---
title: "Road to be an full stack Infrastructure engineer"
date: "2025-11-29"
description: "Learn how to include media files in your blog posts"
---

Week 1: Linux, Git, HTTP Foundations

1. Theory Keywords

* Linux filesystem hierarchy basics
* Shell navigation, permissions, processes
* SSH keys, remote login workflow
* Git init/clone/commit/push/pull basics
* Branching, merging, simple workflows
* HTTP request/response lifecycle
* HTTP methods (GET/POST/PUT/DELETE)
* HTTP status codes families (1xx–5xx)
* REST resource vs representation
* cURL basics for API exploration

2. Hands-on Labs (Inputs → Expected Outputs Only)

* Lab 1 — Navigate Linux and manage files; Inputs: common shell commands, sample dirs; Expected Outputs: created/renamed files, correct permissions; Validation: commands logged in shell history
* Lab 2 — Git local repo + GitHub; Inputs: sample Python project, GitHub remote; Expected Outputs: commits pushed with clear messages; Validation: repo visible on GitHub
* Lab 3 — Call public HTTP APIs with curl; Inputs: public JSON API endpoint; Expected Outputs: JSON responses, headers inspected; Validation: documented requests/responses

3. Mini-Project Idea

* Title: Dev Environment Bootstrapper
* Goal: Automate initial project setup on Linux
* Inputs: shell scripts, git config, editor settings
* Outputs: ready-to-code project folder
* Constraints: idempotent, no manual steps
* Success Metrics: one-command setup, repeatable
* Deliverables: repo with script, README, demo log

4. Q&A Checklist

* Q1: What are key directories in Linux FHS?
* Q2: How does Git track file changes internally?
* Q3: How do HTTP methods map to CRUD operations?
* Q4: How to debug “permission denied” on Linux?
* Q5: Tradeoffs of Gitflow vs trunk-based workflow?

---

Week 2: Python Backend Essentials

1. Theory Keywords

* Python virtual environments (venv/poetry)
* Dependency management, requirements lockfile
* Logging basics, logging levels
* Packaging structure for backend services
* Config via env vars, .env files
* Pydantic models concept
* Type hints for API codebases
* Error handling, exceptions hierarchy
* Basic testing philosophy (unit vs integration)
* Python project layout (src/ tests/)

2. Hands-on Labs

* Lab 1 — Create Python service skeleton; Inputs: venv, src layout; Expected Outputs: runnable main.py; Validation: `python -m` run success
* Lab 2 — Add logging + config; Inputs: env vars, logging module; Expected Outputs: logs with level + context; Validation: config change reflected without code change
* Lab 3 — Write basic unit tests; Inputs: pytest, sample module; Expected Outputs: tests passing in CI-like run; Validation: `pytest` exit code 0

3. Mini-Project Idea

* Title: Configurable Utility Service
* Goal: Build a small CLI service with config + tests
* Inputs: CLI args, env vars, config file
* Outputs: deterministic actions, structured logs
* Constraints: full type hints, basic tests
* Success Metrics: >80% test coverage, clear logs
* Deliverables: repo, test report, usage examples

4. Q&A Checklist

* Q1: Why use virtual environments in backend work?
* Q2: How do type hints help large services?
* Q3: How to externalize configuration safely?
* Q4: How to debug import path issues in Python?
* Q5: Tradeoffs of “monorepo” vs “multi-repo” for backends?

---

Week 3: FastAPI Fundamentals

1. Theory Keywords

* ASGI vs WSGI basics
* FastAPI routing and path operations
* Request body, query, path parameters
* Response models, Pydantic schemas
* Validation errors and 422 responses
* Dependency injection in FastAPI
* Middleware concept in web frameworks
* CORS concepts and configuration
* OpenAPI schema auto-generation
* API documentation via Swagger UI

2. Hands-on Labs

* Lab 1 — Build basic CRUD API with FastAPI; Inputs: in-memory store, Pydantic models; Expected Outputs: CRUD endpoints working; Validation: manual curl tests pass
* Lab 2 — Add DI and middleware; Inputs: dependency functions, logging middleware; Expected Outputs: per-request logs; Validation: logs contain request IDs
* Lab 3 — Explore OpenAPI docs; Inputs: running app; Expected Outputs: working Swagger, proper models; Validation: endpoints invokable from docs

3. Mini-Project Idea

* Title: Simple Task Management API
* Goal: Create a task CRUD API with validation
* Inputs: task schema, in-memory repo, FastAPI
* Outputs: REST endpoints, OpenAPI spec
* Constraints: no DB yet, strict validation
* Success Metrics: full CRUD + filter by status
* Deliverables: running app, API docs, test calls

4. Q&A Checklist

* Q1: How does FastAPI leverage type hints?
* Q2: What are path vs query parameters differences?
* Q3: How does dependency injection simplify cross-cutting logic?
* Q4: How to debug 422 validation errors?
* Q5: Tradeoffs of in-memory vs persistent storage in early stages?

---

Week 4: HTTP, REST, and API Design

1. Theory Keywords

* REST constraints and Richardson Maturity Model
* Resource vs endpoint vs operation
* Idempotency and safe methods
* Pagination patterns (limit/offset, cursor)
* Filtering, sorting, search semantics
* Error handling, RFC7807-style responses
* Versioning strategies (path, header, media type)
* API documentation standards (OpenAPI, JSON Schema)
* Rate limiting and throttling basics
* Backwards compatibility for APIs

2. Hands-on Labs

* Lab 1 — Design resource-oriented endpoints; Inputs: domain model; Expected Outputs: endpoint table; Validation: clear resource URIs
* Lab 2 — Implement pagination and filtering; Inputs: FastAPI service; Expected Outputs: paginated list endpoints; Validation: boundary cases handled
* Lab 3 — Implement consistent error format; Inputs: error handler middleware; Expected Outputs: standardized JSON errors; Validation: all failures use same shape

3. Mini-Project Idea

* Title: Product Catalog API
* Goal: Design a RESTful API for browsing products
* Inputs: product schema, query options, filters
* Outputs: endpoints, pagination, error model
* Constraints: clean resource naming, versioning strategy
* Success Metrics: no breaking changes for new filters
* Deliverables: API design doc, FastAPI implementation

4. Q&A Checklist

* Q1: What makes an API “RESTful”?
* Q2: Why are idempotent methods important?
* Q3: How to design robust pagination for large data?
* Q4: How to debug inconsistent error response formats?
* Q5: Tradeoffs of URL versioning vs header-based versioning?

---

Week 5: Relational DB and SQL Foundations

1. Theory Keywords

* Relational model concepts (tables, keys)
* Normalization basics (1NF, 2NF, 3NF)
* Primary key, foreign key constraints
* Joins (inner, left, right, full)
* Aggregations (GROUP BY, HAVING)
* Indexes, clustered vs non-clustered
* Transactions, ACID properties
* Isolation levels overview
* ORMs vs raw SQL tradeoffs
* Basic PostgreSQL administration

2. Hands-on Labs

* Lab 1 — Design normalized schema; Inputs: simple domain; Expected Outputs: ERD, SQL DDL; Validation: no obvious redundancy
* Lab 2 — Write queries with joins/aggregates; Inputs: sample data; Expected Outputs: correct result sets; Validation: manual checks with expected totals
* Lab 3 — Experiment with indexes; Inputs: large-ish table; Expected Outputs: improved query plans; Validation: EXPLAIN timings before/after

3. Mini-Project Idea

* Title: Orders and Customers Schema
* Goal: Model relational schema for orders domain
* Inputs: customers, orders, line items, payments
* Outputs: DDL scripts, sample queries
* Constraints: referential integrity enforced
* Success Metrics: common queries under target latency
* Deliverables: SQL scripts, ERD diagram, query set

4. Q&A Checklist

* Q1: Why normalize tables and when stop?
* Q2: How do primary and foreign keys relate?
* Q3: How do indexes help and hurt performance?
* Q4: How to debug slow queries systematically?
* Q5: Tradeoffs of ORM abstraction vs hand-written SQL?

---

Week 6: FastAPI + Database Integration

1. Theory Keywords

* SQLAlchemy ORM fundamentals
* Session lifecycle and unit of work
* N+1 query problem and mitigation
* Transactions in application layer
* Connection pooling basics
* Migrations with Alembic
* DTO vs entity vs schema patterns
* Repository pattern for persistence
* Testing with in-memory or test DB
* Error handling for DB-related failures

2. Hands-on Labs

* Lab 1 — Connect FastAPI to PostgreSQL; Inputs: DSN env var, SQLAlchemy; Expected Outputs: basic CRUD persisted; Validation: data visible in DB
* Lab 2 — Implement migrations; Inputs: Alembic config; Expected Outputs: versioned schema changes; Validation: upgrade/downgrade works
* Lab 3 — Fix N+1 queries; Inputs: inefficient endpoints; Expected Outputs: optimized eager loading; Validation: fewer queries logged per request

3. Mini-Project Idea

* Title: Persistent Task Management API
* Goal: Upgrade earlier task API to PostgreSQL
* Inputs: existing FastAPI app, DB schema
* Outputs: persisted tasks, migrations, tests
* Constraints: no data loss across deployments
* Success Metrics: all endpoints < target latency
* Deliverables: code, migration scripts, performance report

4. Q&A Checklist

* Q1: How does SQLAlchemy map objects to tables?
* Q2: Why is N+1 problematic in APIs?
* Q3: How do app-level transactions interact with DB transactions?
* Q4: How to debug connection pool exhaustion?
* Q5: Tradeoffs of repository pattern vs direct ORM usage?

---

Week 7: Docker and Containerization Basics

1. Theory Keywords

* Container vs VM concepts
* Docker images, layers, base images
* Dockerfile instructions (FROM, RUN, CMD, etc.)
* Multi-stage builds for smaller images
* Docker volumes and bind mounts
* Networking: bridge, ports, links basics
* Docker Compose for multi-service setups
* Image tagging and versioning
* Security basics: root vs non-root containers
* Image registries (Docker Hub, ECR, GCR)

2. Hands-on Labs

* Lab 1 — Containerize FastAPI app; Inputs: Dockerfile, app; Expected Outputs: running container; Validation: endpoint reachable on host
* Lab 2 — Add DB via Docker Compose; Inputs: FastAPI + Postgres; Expected Outputs: two containers networked; Validation: API uses DB successfully
* Lab 3 — Optimize image size; Inputs: multi-stage build; Expected Outputs: smaller final image; Validation: image size comparison

3. Mini-Project Idea

* Title: Containerized Dev Stack
* Goal: Run complete backend stack using Docker Compose
* Inputs: app, DB, admin tool (pgAdmin etc.)
* Outputs: one-command `docker-compose up`
* Constraints: env-based config, no local installs
* Success Metrics: new dev onboarding in <30 minutes
* Deliverables: compose file, docs, sample `.env`

4. Q&A Checklist

* Q1: How do containers differ from VMs technically?
* Q2: What is a Docker image layer and why care?
* Q3: How does Docker networking expose services?
* Q4: How to debug container failing to start?
* Q5: Tradeoffs of Docker Compose vs local installs for dev?

---

Week 8: Linux & Networking for DevOps

1. Theory Keywords

* Systemd services, logs, journalctl
* Process management (ps, top, htop)
* File permissions, groups, sudoers
* TCP/IP basics, ports, sockets
* DNS resolution, /etc/hosts
* curl, netcat, telnet for debugging
* firewalls concepts (iptables, ufw)
* Load balancer vs reverse proxy
* SSL/TLS certificates basics
* Common network troubleshooting workflow

2. Hands-on Labs

* Lab 1 — Create systemd service for app; Inputs: unit file, binary; Expected Outputs: app managed by systemd; Validation: enable, start, restart success
* Lab 2 — Debug connectivity issues; Inputs: fake broken configs; Expected Outputs: root cause identified; Validation: connectivity restored
* Lab 3 — Generate self-signed TLS cert; Inputs: openssl; Expected Outputs: cert + key; Validation: service serving HTTPS

3. Mini-Project Idea

* Title: Network Troubleshooting Playbook
* Goal: Build repeatable steps for connectivity debugging
* Inputs: curl, nc, traceroute, logs
* Outputs: documented runbooks, example cases
* Constraints: covers DNS, firewall, routing scenarios
* Success Metrics: reduce mean time to diagnose in exercises
* Deliverables: markdown guide, sample scenarios

4. Q&A Checklist

* Q1: How does DNS resolution flow from host to server?
* Q2: Difference between reverse proxy and load balancer?
* Q3: How to systematically diagnose “connection refused”?
* Q4: How to debug failing systemd service startup?
* Q5: Tradeoffs of using systemd vs container orchestration?

---

Week 9: CI/CD Foundations

1. Theory Keywords

* Continuous Integration principles
* Continuous Delivery vs Deployment
* Pipeline stages (build, test, deploy)
* Git-based workflows for CI (PRs, hooks)
* Artifact repositories (Docker registry, package repo)
* Secrets management in CI pipelines
* Basic branching strategies for CI/CD
* Rollback concepts, blue/green basics
* Build caching for faster pipelines
* Pipeline as code (YAML definitions)

2. Hands-on Labs

* Lab 1 — Set up CI for FastAPI repo; Inputs: Git hosting, CI service; Expected Outputs: build + tests on PR; Validation: failing tests block merge
* Lab 2 — Build and push Docker image in CI; Inputs: Dockerfile, registry credentials; Expected Outputs: tagged image in registry; Validation: image pulled locally
* Lab 3 — Implement simple staging deploy; Inputs: staging env, deploy script; Expected Outputs: auto deploy on main; Validation: health check after deploy

3. Mini-Project Idea

* Title: Minimal CI/CD Pipeline
* Goal: End-to-end pipeline from commit to container deploy
* Inputs: repo, Dockerfile, CI, target host
* Outputs: automated build, test, deploy
* Constraints: on-merge-only deployment, rollback script
* Success Metrics: one-click rollback, reproducible builds
* Deliverables: pipeline config, scripts, architecture diagram

4. Q&A Checklist

* Q1: Why is CI essential for team velocity?
* Q2: How does CD reduce deployment risk?
* Q3: How to secure secrets in CI pipelines?
* Q4: How to debug flaky CI tests?
* Q5: Tradeoffs of monolithic vs per-service pipelines?

---

Week 10: Caching and Performance Basics

1. Theory Keywords

* Read vs write-heavy workload patterns
* Latency, throughput, tail latency
* In-memory cache concepts (Redis)
* Cache-aside, write-through, write-back
* TTL, eviction policies (LRU, LFU)
* Idempotent operations and retries
* HTTP caching headers (ETag, Cache-Control)
* Pagination performance impacts
* Basic load testing concepts
* Performance budget and SLAs

2. Hands-on Labs

* Lab 1 — Add Redis-based cache to API; Inputs: Redis instance, hot endpoints; Expected Outputs: reduced DB load; Validation: metrics before/after
* Lab 2 — Implement cache invalidation strategy; Inputs: update flows; Expected Outputs: correct fresh data; Validation: no stale reads in scenarios
* Lab 3 — Run simple load test; Inputs: locust/k6 tool; Expected Outputs: latency/throughput metrics; Validation: bottlenecks identified

3. Mini-Project Idea

* Title: Cached Product Listing Service
* Goal: Speed up expensive read API with caching
* Inputs: DB-backed API, Redis, load tool
* Outputs: cached responses, metrics dashboard
* Constraints: correct invalidation on writes
* Success Metrics: p95 latency under target threshold
* Deliverables: code, load test scripts, results report

4. Q&A Checklist

* Q1: When is caching harmful instead of helpful?
* Q2: Difference between cache-aside and write-through?
* Q3: How to set TTLs for different resources?
* Q4: How to debug inconsistent data due to cache?
* Q5: Tradeoffs of per-request vs bulk caching strategies?

---

Week 11: Auth Basics and OAuth2 Intro

1. Theory Keywords

* Authentication vs authorization
* Session vs token-based auth
* JWT structure, claims, signatures
* OAuth2 roles (client, resource owner, server)
* OAuth2 grants overview (auth code, client credentials, etc.)
* OIDC ID token vs access token
* Scopes and consent
* Token introspection and revocation
* Secure password storage (hashing)
* Common auth pitfalls (token leakage, CSRF)

2. Hands-on Labs

* Lab 1 — Add basic auth to FastAPI; Inputs: user table, JWT library; Expected Outputs: login, protected endpoints; Validation: unauthorized blocked
* Lab 2 — Integrate OAuth2 Password Grant (local); Inputs: FastAPI OAuth2 support; Expected Outputs: token issuance; Validation: correct scopes enforced
* Lab 3 — Validate and decode JWTs; Inputs: signed tokens; Expected Outputs: parsed claims, validation errors; Validation: invalid tokens rejected

3. Mini-Project Idea

* Title: Auth-Enabled Task API
* Goal: Upgrade task API to support user auth
* Inputs: user model, JWT, password hashing
* Outputs: secure endpoints per user
* Constraints: no plaintext passwords, expiring tokens
* Success Metrics: unauthorized requests correctly rejected
* Deliverables: API, auth docs, example token flows

4. Q&A Checklist

* Q1: How do JWTs provide stateless auth?
* Q2: What are main OAuth2 actors and flows?
* Q3: How does OIDC extend OAuth2?
* Q4: How to debug invalid signature issues with JWTs?
* Q5: Tradeoffs of long-lived vs short-lived tokens?

---

Week 12: Intro to Cloud (AWS or GCP)

1. Theory Keywords

* Regions, zones, availability concepts
* Compute primitives (EC2, GCE)
* Managed DB services (RDS, Cloud SQL)
* Object storage (S3, GCS)
* VPC, subnets, security groups/firewalls
* IAM roles, policies basics
* Load balancers (ALB/ELB, GCLB)
* Managed container services overview (ECS, GKE, EKS)
* Monitoring and logging services (CloudWatch, Stackdriver)
* Cost awareness and tagging

2. Hands-on Labs

* Lab 1 — Provision compute instance; Inputs: AWS/GCP console; Expected Outputs: running VM; Validation: SSH accessible
* Lab 2 — Deploy app to VM; Inputs: container/image or bare app; Expected Outputs: public endpoint; Validation: app reachable via browser
* Lab 3 — Store assets in object storage; Inputs: bucket, files; Expected Outputs: accessible objects; Validation: correct permissions

3. Mini-Project Idea

* Title: Minimal Cloud-Deployed API
* Goal: Run FastAPI service on cloud VM with DB
* Inputs: VM, managed DB, security groups
* Outputs: public HTTPS endpoint
* Constraints: least privilege IAM, basic firewall rules
* Success Metrics: uptime under test, response latency acceptable
* Deliverables: infra diagram, deployment steps, config snippets

4. Q&A Checklist

* Q1: How do regions and zones affect reliability?
* Q2: Why use managed DB instead of self-hosted?
* Q3: How does IAM enforce least privilege access?
* Q4: How to debug connectivity issues between VM and DB?
* Q5: Tradeoffs of VM deployments vs managed container services?

---

### Phase 2 (Weeks 13–24): Intermediate Backend, Kubernetes, Data Engineering Basics

Week 13: Advanced FastAPI Patterns

1. Theory Keywords

* Routers and modular API structure
* Background tasks and async patterns
* Dependency scopes (request, application)
* Custom middleware for cross-cutting concerns
* Exception handlers and global error mapping
* File uploads/downloads
* Streaming responses
* Rate limiting strategies integration
* API versioning in FastAPI
* Performance tuning basics (uvicorn, workers)

2. Hands-on Labs

* Lab 1 — Refactor API into modular routers; Inputs: monolithic app; Expected Outputs: separated modules; Validation: routes unchanged
* Lab 2 — Add custom logging/metrics middleware; Inputs: middleware skeleton; Expected Outputs: per-request metrics; Validation: metrics captured
* Lab 3 — Implement file upload endpoint; Inputs: FastAPI file support; Expected Outputs: stored files; Validation: files retrievable

3. Mini-Project Idea

* Title: Modular Service Skeleton
* Goal: Create reusable FastAPI service template
* Inputs: routers, middleware, DI, config
* Outputs: starter project for future services
* Constraints: clear boundaries, no circular deps
* Success Metrics: new resource added with minimal boilerplate
* Deliverables: template repo, usage guide

4. Q&A Checklist

* Q1: Why modularize API routes and schemas?
* Q2: How do middleware and DI interact?
* Q3: When use streaming responses over full body?
* Q4: How to debug circular imports in FastAPI apps?
* Q5: Tradeoffs of global vs per-route middleware?

---

Week 14: Nginx and Reverse Proxying

1. Theory Keywords

* Nginx as reverse proxy basics
* Upstreams and load balancing algorithms
* SSL termination at Nginx
* Path-based and host-based routing
* Static asset serving vs backend proxy
* Health checks and upstream failure handling
* Rate limiting and request throttling in Nginx
* Caching responses at edge
* Logging format customization
* Hardening Nginx security (headers, TLS)

2. Hands-on Labs

* Lab 1 — Put Nginx before FastAPI; Inputs: Nginx config, app; Expected Outputs: reverse-proxied API; Validation: app hidden behind Nginx
* Lab 2 — Configure SSL termination; Inputs: certs, Nginx; Expected Outputs: HTTPS endpoint; Validation: browser reports secure
* Lab 3 — Implement simple load balancing; Inputs: multiple app instances; Expected Outputs: distributed requests; Validation: logs show balanced traffic

3. Mini-Project Idea

* Title: Secure API Gateway with Nginx
* Goal: Build Nginx layer in front of microservices
* Inputs: multiple backend services, Nginx configs
* Outputs: host/path-based routed gateway
* Constraints: SSL termination, rate limits, security headers
* Success Metrics: zero direct access to backends, consistent logs
* Deliverables: config files, diagrams, test scenarios

4. Q&A Checklist

* Q1: How does a reverse proxy differ from API gateway?
* Q2: What are pros/cons of SSL termination at edge?
* Q3: How do different load balancing algorithms behave?
* Q4: How to debug 502/504 errors through Nginx?
* Q5: Tradeoffs of using Nginx vs managed gateways?

---

Week 15: Kubernetes Fundamentals

1. Theory Keywords

* Kubernetes architecture (master, nodes)
* Pods, Deployments, ReplicaSets
* Services (ClusterIP, NodePort, LoadBalancer)
* Namespaces, labels, selectors
* ConfigMaps and Secrets
* Liveness and readiness probes
* Rolling updates and rollbacks
* Requests and limits (CPU, memory)
* kubectl basics and manifests
* Local clusters (kind, minikube)

2. Hands-on Labs

* Lab 1 — Deploy app to local cluster; Inputs: deployment YAML; Expected Outputs: running pods; Validation: pods healthy
* Lab 2 — Expose service externally; Inputs: Service manifest; Expected Outputs: reachable endpoint; Validation: curl from host
* Lab 3 — Configure probes and resources; Inputs: health endpoints; Expected Outputs: stable rollout; Validation: no downtime during update

3. Mini-Project Idea

* Title: Containerized App on Kubernetes
* Goal: Run FastAPI + DB on Kubernetes cluster
* Inputs: Docker images, manifests, k8s cluster
* Outputs: stable deployment with services
* Constraints: proper probes, resource requests
* Success Metrics: deployment rollouts without downtime
* Deliverables: manifests, deployment notes, diagrams

4. Q&A Checklist

* Q1: Why are pods the basic unit in Kubernetes?
* Q2: How do Services abstract pod IP churn?
* Q3: How do probes impact rollout safety?
* Q4: How to debug CrashLoopBackOff pods?
* Q5: Tradeoffs of resource over-provisioning vs under-provisioning?

---

Week 16: Kubernetes Advanced: Helm and Ingress

1. Theory Keywords

* Helm charts, templates, values
* Release lifecycle, upgrades, rollbacks
* Chart repository structure
* Ingress vs Service vs Gateway
* Ingress controllers (nginx, cloud-specific)
* TLS via Ingress resources
* Environment-specific values management
* DRY infra with shared charts
* Secret management patterns with Helm
* Kustomize vs Helm comparison

2. Hands-on Labs

* Lab 1 — Convert manifests to Helm chart; Inputs: existing YAML; Expected Outputs: reusable chart; Validation: `helm install` works
* Lab 2 — Configure Ingress for API; Inputs: Ingress + controller; Expected Outputs: single DNS entry; Validation: routing verifies
* Lab 3 — Deploy different envs via values; Inputs: dev/stage values; Expected Outputs: multiple releases; Validation: env-specific settings applied

3. Mini-Project Idea

* Title: Helm-Packaged Microservice
* Goal: Package service stack as Helm chart
* Inputs: app, DB, config, Ingress
* Outputs: installable chart with values
* Constraints: environment overrides, secrets externalized
* Success Metrics: new environment created via one Helm command
* Deliverables: chart, README, values examples

4. Q&A Checklist

* Q1: What problems does Helm solve for teams?
* Q2: How does Ingress compare to traditional load balancers?
* Q3: How to structure values for multiple environments?
* Q4: How to debug failed Helm upgrades?
* Q5: Tradeoffs of using Helm vs plain YAML/kustomize?

---

Week 17: Terraform Fundamentals

1. Theory Keywords

* Infrastructure as Code principles
* Terraform providers, resources, data sources
* State files, backends, locking
* Plan and apply lifecycle
* Variables, outputs, locals
* Modules basics and reuse
* Remote state and shared infra data
* Workspaces for environments
* Drift detection and reconciliation
* Secret handling and .tfvars security

2. Hands-on Labs

* Lab 1 — Provision simple infra; Inputs: Terraform, cloud account; Expected Outputs: VM or DB resource; Validation: resource visible in console
* Lab 2 — Create reusable module; Inputs: repeated pattern; Expected Outputs: module used in multiple places; Validation: single module change propagated
* Lab 3 — Configure remote state; Inputs: backend (S3/GCS); Expected Outputs: remote state in bucket; Validation: state lock works

3. Mini-Project Idea

* Title: Terraform Base Infrastructure
* Goal: Define core network + compute stack as code
* Inputs: VPC, subnets, security groups, instances
* Outputs: parameterized Terraform modules
* Constraints: remote state, no manual edits
* Success Metrics: new environment from scratch with one apply
* Deliverables: Terraform repo, diagrams, usage guide

4. Q&A Checklist

* Q1: Why keep infra in code vs console clicks?
* Q2: How does Terraform state impact collaboration?
* Q3: How do modules improve maintainability?
* Q4: How to debug a failing Terraform apply?
* Q5: Tradeoffs of large monolithic Terraform repo vs multiple stacks?

---

Week 18: Terraform Modules and Environments

1. Theory Keywords

* Module composition patterns (root vs child)
* Input/output design for modules
* Environment layering (global, env, app)
* Pattern: network, database, service modules
* Cross-module dependencies via remote state
* Handling breaking changes in modules
* Versioning modules (tags, registries)
* Testing infra changes (plan, sandbox)
* Policy-as-code overview (Sentinel, OPA)
* Multi-account/multi-project setups basics

2. Hands-on Labs

* Lab 1 — Split stack into modules; Inputs: flat config; Expected Outputs: composable modules; Validation: behavior unchanged
* Lab 2 — Implement dev/stage/prod environments; Inputs: workspaces or dirs; Expected Outputs: separate env resources; Validation: no accidental cross-env changes
* Lab 3 — Introduce module versioning; Inputs: git tags; Expected Outputs: locked versions per env; Validation: upgrade path tested

3. Mini-Project Idea

* Title: Enterprise Terraform Module Library
* Goal: Create reusable modules for core building blocks
* Inputs: VPC, database, service, monitoring
* Outputs: documented module library
* Constraints: no env-specific logic in modules
* Success Metrics: new service infra using only modules
* Deliverables: modules, examples, reference architecture

4. Q&A Checklist

* Q1: How to design good module boundaries?
* Q2: Why separate global vs environment state?
* Q3: How to safely roll out breaking module changes?
* Q4: How to debug unintentional resource recreation in plan?
* Q5: Tradeoffs of workspaces vs separate state files?

---

Week 19: Data Engineering Foundations and SQL Mastery

1. Theory Keywords

* OLTP vs OLAP workloads
* Dimensional modeling (facts, dimensions)
* Star vs snowflake schemas
* Window functions and analytic queries
* CTEs for complex SQL
* SCD types overview (0–6)
* Partitioning and clustering large tables
* Data quality constraints and checks
* Query optimization basics in warehouses
* Data ingestion patterns (batch vs stream)

2. Hands-on Labs

* Lab 1 — Design dimensional model; Inputs: analytical use cases; Expected Outputs: fact/dim tables; Validation: core queries mapped
* Lab 2 — Implement window functions; Inputs: warehouse table; Expected Outputs: ranking, rolling metrics; Validation: results match expectations
* Lab 3 — Experiment with partitioning; Inputs: big table; Expected Outputs: improved queries; Validation: performance metrics before/after

3. Mini-Project Idea

* Title: Sales Analytics Warehouse Schema
* Goal: Design and implement analytic schema for sales
* Inputs: orders, customers, products, calendar
* Outputs: facts/dims, example dashboards queries
* Constraints: SCD handling for key dimensions
* Success Metrics: key KPIs derived by SQL only
* Deliverables: DDL, ERD, sample analytic queries

4. Q&A Checklist

* Q1: Why separate OLTP and OLAP systems?
* Q2: What are benefits of star schemas?
* Q3: When use specific SCD types?
* Q4: How to debug incorrect aggregation results?
* Q5: Tradeoffs of wide tables vs normalized models in warehouses?

---

Week 20: Airflow and ETL/ELT Basics

1. Theory Keywords

* DAGs, tasks, dependencies
* Operators and sensors
* Scheduling vs backfilling
* XComs and data passing patterns
* Idempotent ETL tasks
* ETL vs ELT conceptually
* Airflow deployment options
* Monitoring DAG runs, SLAs
* Config, connections, variables
* Common failure and retry strategies

2. Hands-on Labs

* Lab 1 — Create simple DAG; Inputs: Airflow instance; Expected Outputs: scheduled runs; Validation: task success in UI
* Lab 2 — Build ETL from DB to warehouse; Inputs: source DB, warehouse; Expected Outputs: loaded table; Validation: row counts match
* Lab 3 — Implement retries/idempotency; Inputs: flaky tasks; Expected Outputs: correct final state; Validation: multiple runs safe

3. Mini-Project Idea

* Title: Daily Data Pipeline for Orders
* Goal: Build daily incremental load from OLTP to warehouse
* Inputs: source DB, dimensional schema, Airflow
* Outputs: DAGs for load and transform
* Constraints: idempotent, recoverable, monitored
* Success Metrics: consistent data after reruns/retries
* Deliverables: DAG code, runbook, monitoring screenshots

4. Q&A Checklist

* Q1: How do DAGs differ from code-based scripts?
* Q2: Why make ETL tasks idempotent?
* Q3: How to design Airflow retries correctly?
* Q4: How to debug stuck DAG or zombie tasks?
* Q5: Tradeoffs of ETL vs ELT architectures?

---

Week 21: Kafka Fundamentals

1. Theory Keywords

* Topics, partitions, offsets
* Producers, consumers, consumer groups
* At-least-once vs at-most-once vs exactly-once
* Retention policies and compaction
* Ordering guarantees within partitions
* Schema evolution (Avro/Protobuf)
* Backpressure and consumer lag
* Kafka as commit log concept
* Idempotent producers overview
* Stream vs batch integration patterns

2. Hands-on Labs

* Lab 1 — Set up local Kafka cluster; Inputs: docker compose; Expected Outputs: running broker; Validation: health checks
* Lab 2 — Produce and consume messages; Inputs: CLI or client library; Expected Outputs: messages visible; Validation: offsets advancing
* Lab 3 — Experiment with partitions and groups; Inputs: multi-consumer setup; Expected Outputs: parallel consumption; Validation: ordering within partition maintained

3. Mini-Project Idea

* Title: Event Stream for Orders
* Goal: Publish order events for downstream services
* Inputs: order service, Kafka topics, schemas
* Outputs: consistent order-created/updated events
* Constraints: idempotent publishing, schema versioning
* Success Metrics: no message loss in tests, consumer lag controlled
* Deliverables: topic design, producer code, consumer examples

4. Q&A Checklist

* Q1: How does Kafka guarantee ordering?
* Q2: Why use partitions and consumer groups?
* Q3: How do retention and compaction affect storage?
* Q4: How to debug consumer lag and stuck consumers?
* Q5: Tradeoffs of Kafka vs traditional message queues?

---

Week 22: Spark Basics for Data Engineering

1. Theory Keywords

* Spark cluster architecture (driver, executors)
* RDD vs DataFrame vs Dataset
* Lazy evaluation and lineage
* Transformations vs actions
* Partitioning and shuffle operations
* Catalyst optimizer basics
* Spark SQL and DataFrame APIs
* Handling skew and repartitioning
* Integration with Kafka and warehouses
* Job monitoring and tuning basics

2. Hands-on Labs

* Lab 1 — Run Spark job locally; Inputs: sample dataset; Expected Outputs: transformed data; Validation: expected result
* Lab 2 — Write Spark SQL job; Inputs: structured data; Expected Outputs: aggregated table; Validation: results correct
* Lab 3 — Analyze job plan and optimize; Inputs: explain plans; Expected Outputs: fewer shuffles; Validation: improved run time

3. Mini-Project Idea

* Title: Batch Aggregation Pipeline
* Goal: Aggregate order data for daily KPIs using Spark
* Inputs: raw orders, customers, products
* Outputs: aggregated fact tables
* Constraints: partitioned outputs, efficient joins
* Success Metrics: job time within window, data accuracy
* Deliverables: Spark code, configs, performance results

4. Q&A Checklist

* Q1: How does lazy evaluation affect Spark job design?
* Q2: Why prefer DataFrames over RDDs for most workloads?
* Q3: How do shuffles impact performance?
* Q4: How to debug failed Spark jobs in cluster logs?
* Q5: Tradeoffs of Spark vs DB-native transformations?

---

Week 23: Data Warehouses, Data Marts, Lakehouse

1. Theory Keywords

* Traditional data warehouse concepts
* Data marts vs enterprise warehouse
* Lake vs warehouse vs lakehouse
* Columnar storage and compression
* Partitioned tables vs views
* Data governance and catalog (Hive Metastore, Glue)
* Data access patterns for BI tools
* Security and row/column-level access
* Cost optimization in warehouses
* Slowly changing dimensions implementation strategies

2. Hands-on Labs

* Lab 1 — Create data mart in warehouse; Inputs: base tables; Expected Outputs: mart schema; Validation: queries simplified
* Lab 2 — Register tables in catalog; Inputs: metadata tools; Expected Outputs: discoverable datasets; Validation: BI connections use catalog
* Lab 3 — Implement SCD Type 2; Inputs: dimension changes; Expected Outputs: historical tracking; Validation: point-in-time queries correct

3. Mini-Project Idea

* Title: Analytics Lakehouse Prototype
* Goal: Combine data lake storage with warehouse-like tables
* Inputs: object storage, Spark, SQL engine
* Outputs: curated tables, marts for reporting
* Constraints: schema evolution, governance, catalog
* Success Metrics: BI queries fast and consistent
* Deliverables: architecture, DDL, demo queries

4. Q&A Checklist

* Q1: When use data marts vs enterprise warehouse?
* Q2: How does lakehouse differ from classic lake?
* Q3: How to implement SCD Type 2 correctly?
* Q4: How to debug inconsistent metrics across tables?
* Q5: Tradeoffs of centralized vs domain-owned warehouses?

---

Week 24: Data Quality, Lineage, and Governance

1. Theory Keywords

* Data quality dimensions (completeness, accuracy, etc.)
* Data validation frameworks (Great Expectations, dbt tests)
* Lineage tracking concept (table, column)
* Metadata management and catalogs
* PII handling and masking strategies
* Access control and RBAC for data
* Regulatory requirements basics (GDPR-like concepts)
* Data contracts between producers/consumers
* Incident management for data issues
* Observability for data pipelines

2. Hands-on Labs

* Lab 1 — Add data quality checks; Inputs: pipeline outputs; Expected Outputs: failing tests for bad data; Validation: alerts triggered
* Lab 2 — Document lineage for key tables; Inputs: pipeline DAGs; Expected Outputs: lineage graph; Validation: producers/consumers identified
* Lab 3 — Implement PII masking; Inputs: sensitive columns; Expected Outputs: masked/non-prod data; Validation: no PII in test environments

3. Mini-Project Idea

* Title: Data Governance Starter Kit
* Goal: Define processes and tools for quality and lineage
* Inputs: existing pipelines, catalog, quality tools
* Outputs: documented checks, lineage views, policies
* Constraints: minimal friction for engineers
* Success Metrics: data issues detected early in tests
* Deliverables: governance doc, example configs, runbooks

4. Q&A Checklist

* Q1: Why is data quality critical for downstream systems?
* Q2: What is data lineage and why useful?
* Q3: How do data contracts reduce pipeline breakage?
* Q4: How to debug data discrepancy between source and warehouse?
* Q5: Tradeoffs of strict governance vs agility in data teams?

---

### Phase 3 (Weeks 25–36): Distributed Systems, Enterprise Patterns, Reliability & Security

Week 25: Distributed Systems Basics

1. Theory Keywords

* Latency vs bandwidth vs throughput
* CAP theorem and real-world implications
* Consistency models (strong, eventual)
* Idempotency in distributed operations
* Circuit breaker pattern basics
* Retries, backoff, jitter
* Timeouts and deadlines
* Heartbeats and health checks
* Distributed tracing concepts
* Service discovery basics

2. Hands-on Labs

* Lab 1 — Simulate partial failures; Inputs: composed services; Expected Outputs: degraded but working system; Validation: no total outage
* Lab 2 — Implement retries with backoff; Inputs: flaky dependency; Expected Outputs: limited retries; Validation: no retry storms
* Lab 3 — Add simple distributed tracing; Inputs: tracing library; Expected Outputs: trace IDs across services; Validation: end-to-end trace visible

3. Mini-Project Idea

* Title: Resilient Multi-Service Demo
* Goal: Build multi-service app with basic resilience patterns
* Inputs: multiple FastAPI services, network faults
* Outputs: circuit breakers, retries, timeouts
* Constraints: no infinite retries, bounded timeouts
* Success Metrics: graceful degradation under injected faults
* Deliverables: code, chaos scenarios, trace screenshots

4. Q&A Checklist

* Q1: How does CAP theorem guide system design?
* Q2: Why are idempotent operations crucial with retries?
* Q3: How do timeouts and circuit breakers interact?
* Q4: How to debug intermittent failures across services?
* Q5: Tradeoffs of strong vs eventual consistency per use case?

---

Week 26: Microservices and Service Mesh (Istio)

1. Theory Keywords

* Microservices vs monolith tradeoffs
* Service boundaries and bounded contexts
* Service mesh concept and sidecar proxies
* Istio basics: gateway, virtual service, destination rule
* mTLS between services
* Traffic splitting and canary releases
* Policy enforcement (authz, rate limits)
* Observability via mesh (metrics, traces)
* Failure injection and chaos features
* Mesh vs traditional API gateway

2. Hands-on Labs

* Lab 1 — Deploy microservices with Istio; Inputs: Kubernetes cluster; Expected Outputs: mesh-enabled services; Validation: sidecars injected
* Lab 2 — Configure traffic splitting; Inputs: v1/v2 of service; Expected Outputs: weighted traffic; Validation: logs show splits
* Lab 3 — Enable mTLS; Inputs: Istio security configs; Expected Outputs: encrypted service-to-service; Validation: plain requests blocked

3. Mini-Project Idea

* Title: Istio-Based Microservice Platform
* Goal: Run a small microservice system on service mesh
* Inputs: multiple services, Istio configs, gateway
* Outputs: mTLS, traffic policies, observability
* Constraints: no direct service-to-service without mesh
* Success Metrics: safe canary rollout, rich telemetry
* Deliverables: configs, diagrams, rollout playbooks

4. Q&A Checklist

* Q1: When are microservices appropriate vs monolith?
* Q2: How does a service mesh differ from API gateway?
* Q3: How do VirtualServices control traffic flows?
* Q4: How to debug routing issues inside mesh?
* Q5: Tradeoffs of adding service mesh complexity?

---

Week 27: Advanced Auth: OAuth2, OIDC, ADFS, Keycloak

1. Theory Keywords

* OAuth2 Authorization Code Flow (PKCE)
* OIDC discovery and well-known endpoints
* ID token validation best practices
* Roles vs scopes vs claims
* Identity provider vs service provider
* Keycloak concepts (realms, clients, roles)
* ADFS basics and integration patterns
* SSO flows (SAML vs OIDC)
* Token introspection endpoints
* Multi-tenant auth considerations

2. Hands-on Labs

* Lab 1 — Integrate FastAPI with Keycloak; Inputs: Keycloak realm, client; Expected Outputs: login via Keycloak; Validation: tokens validated server-side
* Lab 2 — Implement Authorization Code Flow; Inputs: public client, PKCE; Expected Outputs: secure auth flow; Validation: tokens received only after redirect
* Lab 3 — Map roles/claims to app permissions; Inputs: Keycloak roles; Expected Outputs: RBAC in API; Validation: role-based access enforced

3. Mini-Project Idea

* Title: Enterprise SSO Integration
* Goal: Add SSO to microservice using Keycloak/ADFS
* Inputs: IdP config, OIDC metadata, FastAPI gateway
* Outputs: SSO-protected API endpoints
* Constraints: tokens validated, RBAC enforced, logout handled
* Success Metrics: SSO works end-to-end with minimal local accounts
* Deliverables: config, sequence diagrams, security notes

4. Q&A Checklist

* Q1: How does Authorization Code Flow improve security?
* Q2: What is difference between access and ID tokens?
* Q3: How to design scopes and roles for enterprise apps?
* Q4: How to debug failed token validation and clock skew issues?
* Q5: Tradeoffs of self-hosted Keycloak vs managed IdP?

---

Week 28: CQRS and Event Sourcing

1. Theory Keywords

* CQRS command vs query separation
* Write model vs read model
* Event sourcing basics, event stores
* Event versioning and evolution
* Projections for read models
* Idempotent command handlers
* Consistency models for CQRS systems
* Replay and rebuilding read models
* Snapshots for large event streams
* Testing patterns for event-sourced systems

2. Hands-on Labs

* Lab 1 — Split API into command/query endpoints; Inputs: existing service; Expected Outputs: separate handlers; Validation: no side effects in queries
* Lab 2 — Implement basic event store; Inputs: DB table; Expected Outputs: append-only events; Validation: ordered events per aggregate
* Lab 3 — Build projection for read model; Inputs: events; Expected Outputs: derived table; Validation: projection matches event history

3. Mini-Project Idea

* Title: Event-Sourced Order Service
* Goal: Implement order lifecycle with CQRS and event sourcing
* Inputs: commands, events, event store, read models
* Outputs: command API, query API, projections
* Constraints: idempotent handlers, replayable events
* Success Metrics: full history replays to same state
* Deliverables: architecture doc, code, test scenarios

4. Q&A Checklist

* Q1: Why separate command and query responsibilities?
* Q2: How does event sourcing change persistence design?
* Q3: How to evolve event schemas safely?
* Q4: How to debug projection inconsistencies?
* Q5: Tradeoffs of CQRS/event sourcing vs traditional CRUD?

---

Week 29: Saga Patterns and Distributed Transactions

1. Theory Keywords

* Distributed transaction problem overview
* Orchestration vs choreography sagas
* Compensating actions concept
* Saga state machine modeling
* Idempotent and reversible operations
* Failure handling within sagas
* Timeouts and dead letter queues
* Outbox pattern for reliable publishing
* Testing sagas end-to-end
* Monitoring and tracing sagas

2. Hands-on Labs

* Lab 1 — Implement simple saga orchestrator; Inputs: multi-step business flow; Expected Outputs: successful flows orchestrated; Validation: all services invoked correctly
* Lab 2 — Add compensating actions; Inputs: failing step; Expected Outputs: partial work undone; Validation: consistent final state
* Lab 3 — Integrate saga with Kafka; Inputs: saga events; Expected Outputs: event-driven choreography; Validation: saga completion from logs

3. Mini-Project Idea

* Title: Order Fulfillment Saga
* Goal: Coordinate payment, inventory, shipping services
* Inputs: microservices, Kafka, orchestration logic
* Outputs: resilient multi-step process
* Constraints: compensations for each step, no global locks
* Success Metrics: no stuck orders in failure scenarios
* Deliverables: saga design, sequence diagrams, code

4. Q&A Checklist

* Q1: Why are 2PC transactions hard at scale?
* Q2: Compare saga orchestration vs choreography.
* Q3: How to design compensating actions?
* Q4: How to debug partial failures in sagas?
* Q5: Tradeoffs of eventual consistency in user flows?

---

Week 30: Streaming Pipelines with Kafka + Spark

1. Theory Keywords

* Lambda vs Kappa architectures
* Kafka Streams vs Spark Structured Streaming
* Exactly-once semantics in streams
* Checkpointing and state stores
* Windowed aggregations in streaming
* Late data handling and watermarking
* Backpressure in streaming jobs
* Idempotent sinks and deduplication
* Monitoring streaming pipelines
* Reprocessing and replays

2. Hands-on Labs

* Lab 1 — Build Spark streaming job from Kafka; Inputs: topic, messages; Expected Outputs: processed events; Validation: output tables updated
* Lab 2 — Implement windowed aggregation; Inputs: timestamped events; Expected Outputs: rolling metrics; Validation: windows correct
* Lab 3 — Handle restart and replays; Inputs: job restarts; Expected Outputs: no duplicates; Validation: final state consistent

3. Mini-Project Idea

* Title: Real-Time Order Analytics
* Goal: Process order events into real-time metrics
* Inputs: Kafka order topics, Spark streaming, warehouse
* Outputs: near real-time aggregates
* Constraints: idempotent writes, late event handling
* Success Metrics: latency under SLA, accurate metrics
* Deliverables: streaming job code, configs, dashboards

4. Q&A Checklist

* Q1: How do streaming semantics differ from batch?
* Q2: What is watermarking and why important?
* Q3: How to achieve idempotency in streaming sinks?
* Q4: How to debug lagging or stuck streaming jobs?
* Q5: Tradeoffs of Lambda vs Kappa architectures?

---

Week 31: Data Mesh and Domain-Oriented Data

1. Theory Keywords

* Data mesh principles (domain ownership, etc.)
* Data product concept and SLAs
* Federated computational governance
* Domain-oriented teams owning pipelines
* Self-serve data infra platform idea
* Contract-based data interfaces
* Mesh vs centralized warehouse
* Interoperability across domains
* Cross-domain lineage and catalog
* Organizational impacts of data mesh

2. Hands-on Labs

* Lab 1 — Identify data domains in example org; Inputs: sample business; Expected Outputs: domain map; Validation: clear ownership
* Lab 2 — Design data product API; Inputs: domain use cases; Expected Outputs: schema + SLAs; Validation: consumers’ needs covered
* Lab 3 — Implement sample domain data pipeline; Inputs: domain sources; Expected Outputs: published data product; Validation: described in catalog

3. Mini-Project Idea

* Title: Data Mesh Pilot Design
* Goal: Propose mesh model for two domains
* Inputs: domain boundaries, existing pipelines, tools
* Outputs: data product specs, governance patterns
* Constraints: minimal disruption, clear SLAs
* Success Metrics: consumers can discover and use domain data
* Deliverables: design document, diagrams, product contracts

4. Q&A Checklist

* Q1: What problems does data mesh aim to solve?
* Q2: How do data products differ from tables?
* Q3: How to define SLAs for data products?
* Q4: How to debug cross-domain schema breakages?
* Q5: Tradeoffs of mesh vs centralized data platform?

---

Week 32: Observability: Logging, Metrics, Tracing

1. Theory Keywords

* Three pillars of observability
* Structured logging vs plain text
* Log correlation with trace IDs
* Metrics types (counter, gauge, histogram)
* RED and USE metrics
* Distributed tracing basics (spans, traces)
* OpenTelemetry concepts
* Dashboards and alerting rules
* SLOs and error budgets
* On-call and incident response basics

2. Hands-on Labs

* Lab 1 — Add structured logs to services; Inputs: logging lib; Expected Outputs: JSON logs; Validation: logs queryable by fields
* Lab 2 — Instrument metrics; Inputs: metrics client, endpoints; Expected Outputs: request metrics; Validation: dashboards show trends
* Lab 3 — Enable tracing across services; Inputs: tracing SDK, collector; Expected Outputs: end-to-end traces; Validation: latency breakdown visible

3. Mini-Project Idea

* Title: Observability Stack Integration
* Goal: Integrate logs, metrics, traces for microservices
* Inputs: app, Prometheus/Loki/Jaeger-like stack
* Outputs: unified observability dashboards
* Constraints: low overhead, consistent labels
* Success Metrics: issues diagnosable within minutes via dashboards
* Deliverables: config, dashboards, incident simulation report

4. Q&A Checklist

* Q1: Why is observability different from simple monitoring?
* Q2: How do metrics complement logs and traces?
* Q3: How to choose key SLOs for a service?
* Q4: How to debug high latency using traces?
* Q5: Tradeoffs of sampling vs full tracing?

---

Week 33: Reliability, SRE, and Scaling

1. Theory Keywords

* SRE principles and error budgets
* Capacity planning basics
* Horizontal vs vertical scaling
* Autoscaling in Kubernetes (HPA, VPA)
* Thundering herd problem and mitigation
* Rate limiting and quotas
* Load testing at scale
* Regional redundancy and failover
* Disaster recovery RPO/RTO
* Blameless postmortem practices

2. Hands-on Labs

* Lab 1 — Configure HPA for service; Inputs: metrics; Expected Outputs: auto-scaling; Validation: replicas change under load
* Lab 2 — Run realistic load tests; Inputs: scenario scripts; Expected Outputs: performance curves; Validation: bottlenecks identified
* Lab 3 — Practice failover scenario; Inputs: multi-region or multi-node; Expected Outputs: partial outage recovery; Validation: RTO measured

3. Mini-Project Idea

* Title: Scalability and Reliability Plan
* Goal: Define scaling and SRE practices for chosen system
* Inputs: traffic patterns, SLAs, infra options
* Outputs: scaling strategy, SLOs, runbooks
* Constraints: budget-aware, simple initial setup
* Success Metrics: passes defined load and failure tests
* Deliverables: plan document, test results, postmortem example

4. Q&A Checklist

* Q1: How do error budgets influence release pace?
* Q2: When scale horizontally vs vertically?
* Q3: How to design effective autoscaling signals?
* Q4: How to debug cascading failures during traffic spikes?
* Q5: Tradeoffs of active-active vs active-passive DR?

---

Week 34: Security for Backend and DevOps

1. Theory Keywords

* Secure coding basics (OWASP Top 10)
* Input validation and output encoding
* SQL injection and ORM safeguards
* CSRF, XSS, clickjacking protection
* Secrets management (vaults, KMS)
* TLS everywhere and certificate rotation
* Container security (image scanning, minimal base)
* Kubernetes security (RBAC, PSP/PodSecurity)
* Network segmentation and zero trust basics
* Security logging and incident response

2. Hands-on Labs

* Lab 1 — Harden FastAPI endpoints; Inputs: insecure app; Expected Outputs: mitigated OWASP issues; Validation: basic scans pass
* Lab 2 — Centralize secrets management; Inputs: vault/KMS; Expected Outputs: no secrets in code; Validation: rotation without redeploy
* Lab 3 — Lock down Kubernetes cluster; Inputs: RBAC, policies; Expected Outputs: restricted access; Validation: unauthorized actions blocked

3. Mini-Project Idea

* Title: Security Hardening Playbook
* Goal: Create checklist and configs to harden services
* Inputs: app, infra, CI/CD
* Outputs: security baselines and automation
* Constraints: minimal friction for developers
* Success Metrics: fewer high-severity findings in scans
* Deliverables: playbook doc, config examples, scan reports

4. Q&A Checklist

* Q1: Which OWASP risks are most relevant to APIs?
* Q2: How does secrets sprawl happen and how to prevent it?
* Q3: How to design secure defaults in services?
* Q4: How to debug failed TLS handshakes in production?
* Q5: Tradeoffs of strict security vs developer velocity?

---

Week 35: End-to-End Enterprise Architecture

1. Theory Keywords

* Layered architecture vs hexagonal architecture
* API gateway patterns in microservices
* Backend-for-frontend (BFF) pattern
* Event-driven architecture and integration
* Command, query, event segregation
* Cross-cutting concerns (auth, logging, metrics)
* Shared libraries vs duplicated logic
* Governance for APIs and services
* Reference architectures for enterprise systems
* Documentation for large systems (C4 model)

2. Hands-on Labs

* Lab 1 — Model system with C4 diagrams; Inputs: chosen domain; Expected Outputs: context, container, component diagrams; Validation: stakeholders understand design
* Lab 2 — Implement API gateway or BFF; Inputs: Nginx or gateway tool; Expected Outputs: unified entry point; Validation: frontends use gateway
* Lab 3 — Introduce event-driven integration; Inputs: Kafka; Expected Outputs: service decoupling; Validation: services communicate via events

3. Mini-Project Idea

* Title: Enterprise Reference System Blueprint
* Goal: Design architecture for mid-large enterprise platform
* Inputs: all prior topics (APIs, mesh, data, auth)
* Outputs: architecture, patterns, integration strategy
* Constraints: scalable, secure, observable, evolvable
* Success Metrics: addresses key NFRs explicitly
* Deliverables: architecture doc, diagrams, pattern catalog

4. Q&A Checklist

* Q1: How to choose between monolith, modular monolith, microservices?
* Q2: When use BFF pattern vs single API?
* Q3: How to enforce cross-cutting policies consistently?
* Q4: How to debug issues in complex, event-driven architectures?
* Q5: Tradeoffs of shared libraries vs independently evolved services?

---

Week 36: Capstone Week: Integrated Backend + DevOps + Data Eng

1. Theory Keywords

* End-to-end SDLC in enterprise contexts
* Cross-team contracts (API, data, SLAs)
* Platform engineering and golden paths
* Cost, performance, reliability tradeoffs
* Tooling consolidation vs sprawl
* Build-vs-buy considerations
* Measuring engineering productivity
* Technical debt management strategies
* Roadmapping and incremental delivery
* Post-incident learning and continuous improvement

2. Hands-on Labs

* Lab 1 — Integrate CI/CD, infra, app, data pipeline; Inputs: prior projects; Expected Outputs: working end-to-end system; Validation: scenario walk-through
* Lab 2 — Run chaos and performance scenarios; Inputs: load + failure tools; Expected Outputs: observed behavior; Validation: system meets or misses targets
* Lab 3 — Run full incident simulation; Inputs: injected issues; Expected Outputs: detection, response, learning; Validation: documented postmortem

3. Mini-Project Idea

* Title: Enterprise Platform Capstone
* Goal: Build or design a complete platform combining backend, DevOps, and data engineering skills
* Inputs: microservices, Kafka, Spark, Airflow, K8s, Terraform, Keycloak, observability stack
* Outputs: coherent architecture and implementation slices
* Constraints: realistic scope, clear tradeoffs, production-minded design
* Success Metrics: demonstrates mastery of core patterns and tools
* Deliverables: code repos, infra configs, architecture doc, demo walkthrough

4. Q&A Checklist

* Q1: How would you explain your architecture to a CTO?
* Q2: Where are the main risks and bottlenecks in your design?
* Q3: How would you evolve the platform over the next year?
* Q4: How would you debug a major outage across app + infra + data?
* Q5: Which tradeoffs did you consciously make and why?

---
