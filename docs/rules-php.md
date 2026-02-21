# RULES.MD - PHP 8.4 Development Standards

**Target**: PHP 8.4+ on WP Engine (Standalone and WordPress-integrated).
**Scope**: This document governs all code generation, architecture, and quality assurance for PHP projects.

---

## PART 1: FIRST PRINCIPLES

These principles take precedence over all other rules.

### P1: Fix Over Create
- **Refactor First**: Always update/fix existing classes rather than creating new files.
- **Complexity Cap**: When `phpstan` or static analysis indicates high cyclomatic complexity (CC > 10), refactor into smaller, single-responsibility classes or pure functions.
- **Service Extraction**: If a class exceeds 200 lines, extract logic into dedicated Services, Repositories, or Value Objects.

### P2: Reusable Testing Infrastructure
- **NO One-Off Scripts**: Do not write diagnostic PHP scripts in the root directory.
- **Composer Scripts as Source of Truth**: All quality commands live in `composer.json` scripts section.
- **Unit vs Integration**: Pure logic tests belong in `tests/Unit/`. WordPress/Database tests belong in `tests/Feature/` (require `wp-env`).

### P3: Documentation Location
- ALL documentation goes in `docs/`.
- **Manifest is Documentation**: `composer.json` must include complete package metadata (description, keywords, license, authors).
- **Self-Documenting Types**: PHP 8.4 type declarations and PHPDoc blocks serve as the primary documentation for data structures.

### P4: Never Defer Necessary Work
- **Strict Types Always**: Every PHP file must begin with `declare(strict_types=1);`. No exceptions.
- **No Type Suppression**: Never use `@phpstan-ignore` or `/** @var */` to silence type errors; fix the type definition.
- **Async Hygiene**: All Promises/async operations (Guzzle, ReactPHP) must be properly awaited or handled.

### P5: Class/Module Isolation
- **Service Purity**: Services must be stateless. Never store request-specific data in class properties.
- **Dependency Injection**: All dependencies must be injected via constructor. No `new` keywords inside business logic.
- **WordPress Globals**: Never access `$wpdb`, `$post`, or other globals directly in domain logic. Wrap in repository classes.

---

## PART 2: CORE STANDARDS

### 2.1 Project Structure

| Requirement | Rule |
|-------------|------|
| **File Markers** | Start files with `declare(strict_types=1);` followed by namespace |
| **Imports** | Absolute namespaces only. No relative `use` statements. Group by type (PHP, Vendor, App). |
| **Entry Points** | `public/index.php` (Web), `bin/console` (CLI), or plugin main file (WordPress). |
| **Complexity** | No file exceeds 250 LOC. Classes must satisfy Single Responsibility. |

**Required Files:**
- `composer.json` (Project manifest and autoloading)
- `phpstan.neon` (Static analysis configuration)
- `phpunit.xml` or `pest.php` (Testing configuration)
- `.env.example` (Environment variable template)

**Canonical Structure:**
```
project-root/
├── .env.example          # Environment template (COMMIT)
├── .env                  # Actual secrets (GITIGNORE)
├── composer.json         # Dependencies and autoloading
├── composer.lock         # Locked versions (COMMIT)
├── phpstan.neon          # Static analysis config
├── rector.php            # Automated refactoring config
├── public/               # Web root (only index.php exposed)
│   └── index.php         # Front controller
├── src/                  # Application source (PSR-4: "App\\")
│   ├── Controllers/
│   ├── Domain/
│   ├── Services/
│   └── Repositories/
├── tests/
│   ├── Unit/             # Isolated unit tests
│   └── Feature/          # Integration tests (require DB)
└── vendor/               # Dependencies (GITIGNORE)
```

### 2.2 Configuration

- **Use `.env` Files**: Use `vlucas/phpdotenv` for environment configuration.
- **Never Commit Secrets**: `.env` must be in `.gitignore`. Provide `.env.example` as template.
- **Never Overwrite**: Existing tokens/keys/passwords must never be overwritten programmatically.
- **Runtime Validation**: All configuration must be validated into typed Config classes at bootstrap.
- **No Hardcoding**: Never hardcode API keys, URLs, or credentials in source files.

### 2.3 Command-Line Interface

- Use **Symfony Console** or **Laravel Zero** for CLI applications.
- Use **WP-CLI** for WordPress-specific commands.
- `composer.json` scripts are the primary command interface:

```json
"scripts": {
    "test": "vendor/bin/pest",
    "test:coverage": "vendor/bin/pest --coverage",
    "analyze": "vendor/bin/phpstan analyse",
    "format": "vendor/bin/pint",
    "lint": "vendor/bin/pint --test",
    "rector": "vendor/bin/rector process --dry-run",
    "ci": ["@lint", "@analyze", "@test"]
}
```

### 2.4 Code Quality Principles

| Principle | Meaning |
|-----------|---------|
| **DRY** | Don't Repeat Yourself (Extract to Services/Traits) |
| **SPOT** | Single Point of Truth (One canonical location for each piece of data) |
| **SOLID** | Single responsibility, Open/closed, Liskov, Interface segregation, Dependency inversion |
| **GRASP** | General Responsibility Assignment Software Patterns |
| **YAGNI** | You Aren't Gonna Need It (No speculative features) |

**Rules:**
- PSR-12 coding style enforced via **Laravel Pint**
- Max two levels of nesting in methods
- Clear, descriptive, unambiguous names (no abbreviations)
- Single responsibility per method/class
- PHPDoc only when types cannot express the contract

### 2.5 Environment & Tooling

| Tool | Purpose |
|------|---------|
| **PHP** | 8.4+ required |
| **Composer** | Package/autoload management |
| **Laravel Pint** | Code formatting (PSR-12) |
| **PHPStan** | Static analysis (Level 6 minimum) |
| **Pest** | Testing framework (preferred over PHPUnit) |
| **Rector** | Automated refactoring and upgrades |

**Composer commands:**
```bash
composer install              # Install dependencies
composer require <package>    # Add dependency
composer require --dev <pkg>  # Add dev dependency
composer dump-autoload        # Regenerate autoloader
composer ci                   # Run full CI pipeline
```

---

## PART 3: QUALITY GATE CHECKLIST

### Tier 1 - Gate Checks (Must Pass Before Commit)

| Tool | Command | Purpose |
|------|---------|---------|
| **Pint** | `composer lint` | Code style enforcement (PSR-12) |
| **PHPStan** | `composer analyze` | Static type analysis (Level 6+) |
| **Pest** | `composer test` | Unit tests |
| **PHP Lint** | `php -l src/` | Syntax validation |

### Tier 2 - Quality Analysis

| Tool | Command | Purpose |
|------|---------|---------|
| **Rector** | `vendor/bin/rector --dry-run` | Identify refactoring opportunities |
| **PHPMD** | `vendor/bin/phpmd src/ text cleancode` | Mess detection |
| **Psalm** | `vendor/bin/psalm` | Alternative static analysis |
| **Composer Audit** | `composer audit` | Security vulnerability scanning |

### Tier 3 - Advanced

| Tool | Command | Purpose |
|------|---------|---------|
| **Infection** | `vendor/bin/infection` | Mutation testing |
| **PHPBench** | `vendor/bin/phpbench run` | Performance benchmarking |
| **Deptrac** | `vendor/bin/deptrac` | Architecture dependency analysis |
| **PHP Insights** | `vendor/bin/phpinsights` | Unified code quality metrics |

### Specialized Tools

| Tool | Purpose |
|------|---------|
| **Mockery** | Test mocking library |
| **Faker** | Test data generation |
| **PHPStan Extensions** | WordPress, Doctrine, Symfony type stubs |
| **Xdebug** | Step debugging and profiling |
| **Psysh** | Interactive REPL (superior to `php -a`) |
| **wp-env** | Docker-based WordPress testing environment |

---

## PART 4: 8-DIMENSION QA FRAMEWORK

Evaluate code on these 8 dimensions (The "PHP Context"):

| Dimension | Question |
|-----------|----------|
| **Type Safety** | Are all inputs/outputs strictly typed? Is `declare(strict_types=1)` present? |
| **SQL Injection** | Are all queries using prepared statements (`$wpdb->prepare()` or PDO)? |
| **XSS Prevention** | Is all output escaped (`esc_html()`, `esc_attr()`, `esc_url()`)? |
| **Dependency Injection** | Are dependencies injected, not instantiated internally? |
| **Global State** | Does this access WordPress globals directly? (Should use repositories) |
| **Serialization** | Is `unserialize()` ever used on user input? (CRITICAL vulnerability) |
| **Error Handling** | Are exceptions caught and logged appropriately? No silent failures? |
| **WP Engine Compat** | Does this respect query limits (1024 char), cache layers, banned functions? |

---

## PART 5: IMPLEMENTATION STANDARDS

### 5.1 Type Hints (PHP 8.4 Syntax)

**Modern Syntax (USE THIS):**
```php
<?php
declare(strict_types=1);

namespace App\Services;

// Explicit return types always
function processItems(array $items): array
{
    $result = [];
    return $result;
}

// Union types (PHP 8.0+)
function fetch(string $url): string|null
{
    // ...
}

// Intersection types (PHP 8.1+)
function process(Countable&Iterator $collection): int
{
    return count($collection);
}

// Constructor Property Promotion with readonly (PHP 8.1+)
final class PaymentService
{
    public function __construct(
        private readonly PaymentRepository $repository,
        private readonly LoggerInterface $logger,
    ) {}
}

// Property Hooks (PHP 8.4)
class User
{
    public string $fullName {
        get => $this->first . ' ' . $this->last;
        set {
            [$this->first, $this->last] = explode(' ', $value, 2);
        }
    }
}

// Asymmetric Visibility (PHP 8.4)
class Product
{
    public private(set) string $sku;      // Public read, private write
    public protected(set) float $price;   // Public read, protected write
}
```

**Anti-Patterns (AVOID):**
```php
// DON'T USE:
$data = ...;                    // Missing type - always declare
function foo($bar) { }          // Missing parameter and return types
/** @var string $x */           // Don't use PHPDoc to fix type errors
mixed $anything;                // Avoid 'mixed' - be explicit
== and !=                       // ALWAYS use === and !== (strict equality)
$$variable                      // Variable variables - STRICTLY PROHIBITED
```

### 5.2 Core Libraries

| Domain | Library | Rule |
|--------|---------|------|
| **Database** | Doctrine DBAL or `$wpdb` | All queries via prepared statements. No raw SQL string interpolation. |
| **HTTP Client** | Guzzle or `wp_remote_*` | Wrapper required for error handling. Set timeouts explicitly. |
| **Validation** | Respect/Validation or Symfony Validator | All external input must be validated before use. |
| **Serialization** | `json_encode()`/`json_decode()` | NEVER use `serialize()`/`unserialize()` on user data. |
| **Logging** | Monolog (PSR-3) | Use `LoggerInterface` type hints for swappable implementations. |
| **Templating** | Blade or Twig | Automatic escaping. Never echo raw HTML in PHP. |

### 5.3 Logging & Error Handling

```php
<?php
declare(strict_types=1);

use Psr\Log\LoggerInterface;

final class PaymentProcessor
{
    public function __construct(
        private readonly LoggerInterface $logger,
    ) {}

    public function process(Payment $payment): void
    {
        $this->logger->info('Processing payment', ['id' => $payment->id]);
        
        try {
            // Business logic
        } catch (PaymentException $e) {
            $this->logger->error('Payment failed', [
                'id' => $payment->id,
                'error' => $e->getMessage(),
            ]);
            throw $e;
        }
    }
}
```

**Rules:**
- Define custom project-specific exceptions extending base Exception classes
- Convert PHP warnings/notices to exceptions at bootstrap:
  ```php
  set_error_handler(function ($severity, $message, $file, $line) {
      throw new ErrorException($message, 0, $severity, $file, $line);
  });
  ```
- Log level must be configurable via environment
- Never catch `\Throwable` silently - always log or rethrow

### 5.4 Testing

**Pest Test Categories:**
```php
// tests/Unit/Services/PaymentServiceTest.php
<?php
declare(strict_types=1);

use App\Services\PaymentService;

describe('PaymentService', function () {
    beforeEach(function () {
        $this->repository = mock(PaymentRepository::class);
        $this->service = new PaymentService($this->repository);
    });

    test('processes valid payment', function () {
        $this->repository
            ->shouldReceive('save')
            ->once()
            ->andReturn(true);

        $result = $this->service->process(new Payment(100));

        expect($result)->toBeTrue();
    });

    test('rejects negative amount', function () {
        expect(fn() => $this->service->process(new Payment(-50)))
            ->toThrow(InvalidArgumentException::class);
    });
});
```

**Test Categories:**
- **Unit**: Pure PHP logic, mocked dependencies, no I/O
- **Feature/Integration**: Requires database, uses `wp-env` for WordPress
- **Architecture**: Deptrac rules for dependency direction

**Critical Rule:** When tests produce no output, assume complete failure. Never assume success from silence.

### 5.5 WordPress-Specific Standards

**Hook Registration (Observer Pattern):**
```php
<?php
declare(strict_types=1);

namespace App\WordPress;

final class PostTypeRegistrar
{
    public function register(): void
    {
        add_action('init', [$this, 'registerQuizPostType']);
    }

    public function registerQuizPostType(): void
    {
        register_post_type('quiz', [
            'public' => true,
            'label' => 'Quizzes',
            'supports' => ['title', 'editor', 'thumbnail'],
        ]);
    }
}
```

**Security Functions (MANDATORY):**
```php
// SANITIZE on input (before database)
$title = sanitize_text_field($_POST['title']);
$email = sanitize_email($_POST['email']);
$id = absint($_GET['id']);

// ESCAPE on output (before display)
echo esc_html($title);                    // Inside HTML tags
echo esc_attr($class);                    // Inside HTML attributes
echo esc_url($link);                      // href/src attributes
echo esc_js($data);                       // Inside JavaScript

// PREPARED STATEMENTS for queries (ALWAYS)
$wpdb->query($wpdb->prepare(
    "SELECT * FROM {$wpdb->posts} WHERE ID = %d",
    $post_id
));

// NONCES for forms (CSRF protection)
wp_nonce_field('my_action', '_wpnonce');
if (!wp_verify_nonce($_POST['_wpnonce'], 'my_action')) {
    wp_die('Security check failed');
}
```

### 5.6 Progress & Reporting

- Use **Symfony Console** progress bars for CLI operations with > 5 steps
- Use transients or object cache for long-running background processes
- Reports: Generate HTML with Tailwind CSS, client-side rendering only

---

## PART 6: VUW METHODOLOGY

**Verifiable Units of Work** - Micro-plans for disciplined PHP development.

### Core Principles

1. **Extreme Granularity**: One class or one specific error per VUW
2. **Verification = Done**: Task incomplete until checklist passes
3. **Sequential Execution**: One VUW at a time; complete before next
4. **Clarity Over Conciseness**: Literal instructions, assume nothing

### VUW Template

```markdown
**VUW_ID:** [e.g., PHP-FIX-001]

**Objective:** [One sentence explaining WHY this matters]

**Files:** [List of files to modify]

**Pre-Work Checkpoint:** git commit before any changes

**Steps:**
1. [Literal instruction with exact code/paths]
2. [Show changes as git diff format]
3. [...]

**Verification:**
- [ ] `composer lint` reports zero errors
- [ ] `composer analyze` reports zero errors (PHPStan Level 6+)
- [ ] `composer test` all tests pass
- [ ] `php -l` syntax check passes on modified files

**Post-Work Checkpoint:** git commit after verification passes
```

### Campaign Structure

Organize VUWs into campaigns by priority:

1. **Application Stability** - Fix blockers preventing `pest` from running
2. **Type Safety** - Achieve zero `phpstan` errors at Level 6+
3. **Code Quality** - Achieve zero `pint` formatting violations
4. **Security Audit** - Pass `composer audit` with no vulnerabilities

---

## PHP 8.4 Features Reference

| Feature | Usage |
|---------|-------|
| `declare(strict_types=1)` | Enforce type contracts at runtime (MANDATORY) |
| Property Hooks | `get`/`set` blocks for computed properties |
| Asymmetric Visibility | `public private(set)` for read-only public access |
| Constructor Promotion | `public readonly Type $prop` in constructor signature |
| `readonly` classes | Immutable value objects |
| `match` expressions | Type-safe switch replacement |
| Named arguments | `func(param: value)` for clarity |
| Attributes | `#[Route('/path')]` for metadata |
| Enums | Native enumeration types |
| Fibers | Cooperative multitasking (advanced) |
| `#[\Deprecated]` | Mark code for deprecation with static analysis support |
| `new` in initializers | Default parameter values can be object instantiations |

---

## WP Engine Constraints Reference

| Constraint | Limit | Mitigation |
|------------|-------|------------|
| **Query Length** | 1024 characters max | Break complex queries; cache IDs |
| **Object Cache** | 1MB per key | Chunk large data; handle cache misses |
| **Banned Functions** | `exec()`, `system()`, `phpinfo()` | Use WP-Cron for background jobs |
| **Banned Plugins** | Caching, Backup plugins | Use platform-provided features |
| **Page Cache** | Varnish for anonymous users | Don't use `$_SESSION`; use cookies |
| **PHP Extensions** | bcmath, curl, gd, imagick, intl, mbstring, mysqli | No `pcntl` (process control) |

---

## Anti-Patterns (NEVER DO)

- **Loose Equality**: Using `==` or `!=` (ALWAYS use `===` and `!==`)
- **Type Juggling Reliance**: Depending on PHP's implicit type coercion
- **Variable Variables**: `$$var` syntax (impossible to analyze statically)
- **Global State**: Accessing `$wpdb`, `$post` directly in domain logic
- **Object Injection**: Using `unserialize()` on user-controlled data
- **Missing `strict_types`**: Files without `declare(strict_types=1);`
- **Hardcoded Credentials**: API keys or passwords in source code
- **Raw SQL**: String interpolation in queries instead of prepared statements
- **Silent Failures**: Empty catch blocks or `@` error suppression
- **Magic Methods Abuse**: Overusing `__get`, `__set`, `__call` (use Property Hooks)
- **God Classes**: Single class handling multiple responsibilities
- **`query_posts()`**: Overwrites main loop (use `WP_Query` or `pre_get_posts`)
- **Direct DB Queries**: Bypassing `$wpdb->prepare()` with raw MySQL
- **Mixing Async Patterns**: Combining callbacks with Promises (pick one)
- **Deprecated Imports**: Using `\Traversable` instead of specific interfaces
