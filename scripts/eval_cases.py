"""Live eval cases for the fast subagent: starting repos, briefs, hidden checks and reference solutions.

Importable without side effects. Every check and hidden command uses python3 and the stdlib only.
A case's hidden files are written into the workdir only after the run, so the model never sees
them; the hidden command exits 0 when the work is correct. `reference` turns the starting repo into
a correct solution and is used only by the offline tests.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Union

CATEGORIES = (
    "spec_module", "follow_pattern", "rename_across_files", "write_tests",
    "fix_failing_test", "large_file_edit", "add_types_docs", "debug_symptom",
)

# Written into every starting repo, so bytecode from running tests does not count as a changed file.
COMMON_FILES = {".gitignore": "__pycache__/\n*.pyc\n"}

UNITTEST_TESTS_DIR = "python3 -m unittest discover -q -s tests -t ."


@dataclass
class Hidden:
    files: Dict[str, str]
    command: str


@dataclass
class Case:
    name: str
    category: str
    task: str
    check: str
    files: Dict[str, str]
    hint_files: List[str]
    hidden: Hidden
    # path -> new content, or a function(old content) -> new content
    reference: Dict[str, Union[str, Callable[[str], str]]]
    limits: Dict[str, int] = field(default_factory=dict)


def starting_files(case: Case) -> Dict[str, str]:
    files = dict(COMMON_FILES)
    files.update(case.files)
    return files


def write_files(root, files: Dict[str, str]) -> None:
    for rel, text in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def apply_reference(case: Case, root) -> None:
    for rel, ref in case.reference.items():
        path = Path(root) / rel
        if callable(ref):
            ref = ref(path.read_text(encoding="utf-8"))
        write_files(root, {rel: ref})


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_once(old: str, new: str) -> Callable[[str], str]:
    def apply(text: str) -> str:
        if text.count(old) != 1:
            raise ValueError("reference anchor not found exactly once")
        return text.replace(old, new)
    return apply


_MODEL_TESTS_GUARD = r'''

import ast as _ast
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile

MODEL_TEST_FILE = @FILE@
MODEL_TEST_MODULE = @MODULE@
ORIGINAL_TEST_SOURCE = @ORIGINAL_TESTS@
ORIGINAL_CODE = @ORIGINAL_CODE@


def _test_names(source):
    """(class, method) names of the test methods defined in test source."""
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return set()
    return {(node.name, item.name) for node in tree.body if isinstance(node, _ast.ClassDef)
            for item in node.body
            if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and item.name.startswith("test")}


def _run_model_tests(replacements):
    """Run the model's test module in a copy of the repo with replacements (path -> source) swapped in."""
    tmp = _tempfile.mkdtemp()
    try:
        copy = _os.path.join(tmp, "repo")
        _shutil.copytree(".", copy, ignore=_shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", "eval_hidden_*"))
        for rel, source in replacements.items():
            with open(_os.path.join(copy, rel), "w", encoding="utf-8") as f:
                f.write(source)
        env = dict(_os.environ, PYTHONDONTWRITEBYTECODE="1")
        return _subprocess.run([_sys.executable, "-m", "unittest", "-q", MODEL_TEST_MODULE], cwd=copy,
                               env=env, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                               universal_newlines=True, timeout=60)
    finally:
        _shutil.rmtree(tmp, ignore_errors=True)


class ModelTestsHidden(unittest.TestCase):
    """The brief asked for tests: they must exist, pass, and fail on the original code."""

    def test_new_tests_were_added(self):
        try:
            with open(MODEL_TEST_FILE, encoding="utf-8") as f:
                current = f.read()
        except OSError:
            current = ""
        new = _test_names(current) - _test_names(ORIGINAL_TEST_SOURCE)
        self.assertTrue(new, "no new test methods in %s" % MODEL_TEST_FILE)

    def test_model_tests_pass(self):
        result = _run_model_tests({})
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])

    def test_model_tests_fail_on_original_code(self):
        result = _run_model_tests(ORIGINAL_CODE)
        self.assertNotEqual(result.returncode, 0, "the tests in %s pass on the original code"
                            % MODEL_TEST_FILE)
'''


def _with_model_tests_guard(hidden: str, test_file: str, original_tests: str,
                            original_code: Dict[str, str]) -> str:
    """hidden plus ModelTestsHidden for test_file; original_code is what the new tests must fail on."""
    module = test_file[:-len(".py")].replace("/", ".")
    guard = (_MODEL_TESTS_GUARD.replace("@FILE@", repr(test_file)).replace("@MODULE@", repr(module))
             .replace("@ORIGINAL_TESTS@", repr(original_tests))
             .replace("@ORIGINAL_CODE@", repr(original_code)))
    main = '\n\nif __name__ == "__main__":'
    if hidden.count(main) != 1:
        raise ValueError("hidden check has no single main block")
    return hidden.replace(main, guard.rstrip("\n") + main)


# =============================================================================================
# 1. spec_module
# =============================================================================================

SPEC_TASK = """Create textutil.py with a function slugify(text, max_length=None) that follows this spec exactly, and test_textutil.py (next to it) with unittest tests for it.

Spec for slugify(text, max_length=None) -> str:
1. Normalize text with unicodedata.normalize("NFKD", text), then drop every non-ASCII character. So "\u00e9" becomes "e", the ligature "\ufb01" becomes "fi", and "\u00df" (which has no ASCII form) is dropped.
2. Lowercase the result.
3. Replace every run of one or more characters that are not a-z or 0-9 with a single hyphen "-". Spaces, underscores, punctuation and existing hyphens all count.
4. Strip leading and trailing hyphens.
5. If max_length is given and the slug is longer than max_length, shorten it at a hyphen boundary: keep the longest run of whole leading words (the parts between hyphens) whose joined length, hyphens included, is at most max_length. If even the first word is longer than max_length, keep only its first max_length characters. The result never ends with a hyphen.
6. If nothing is left, return "".
7. Raise ValueError if max_length is less than 1 (for example 0 or -3). max_length=None means no limit.

Examples: slugify("Hello, World!") == "hello-world"; slugify("  Cr\u00e8me Br\u00fbl\u00e9e ") == "creme-brulee"; slugify("hello-world-foo", max_length=12) == "hello-world"; slugify("hello", max_length=3) == "hel"; slugify("!!!") == "".

Stdlib only, Python 3.9+. Do not add other files. Done means textutil.py and test_textutil.py exist and the check command passes."""

SPEC_README = """# notes-site

Static site generator for a personal notes collection. Python 3.9+, stdlib only.
Modules live in the repo root; tests use unittest and sit next to the module as test_<module>.py.
"""

SPEC_HIDDEN = r'''import unittest

from textutil import slugify


class SlugifySpec(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertEqual(slugify("ABC123def"), "abc123def")
        self.assertEqual(slugify("Top 10 Tips"), "top-10-tips")

    def test_accents_folded(self):
        self.assertEqual(slugify("  Cr\u00e8me Br\u00fbl\u00e9e "), "creme-brulee")
        self.assertEqual(slugify("\u00c5ngstr\u00f6m"), "angstrom")

    def test_nfkd_ligature(self):
        self.assertEqual(slugify("\ufb01le Name"), "file-name")

    def test_non_ascii_dropped(self):
        self.assertEqual(slugify("Stra\u00dfe"), "strae")
        self.assertEqual(slugify("\u65e5\u672c\u8a9e"), "")

    def test_separators_collapse(self):
        self.assertEqual(slugify("foo_bar--baz"), "foo-bar-baz")
        self.assertEqual(slugify("a\tb\nc"), "a-b-c")
        self.assertEqual(slugify("C++ & C#"), "c-c")

    def test_strip_and_empty(self):
        self.assertEqual(slugify("--Hi--"), "hi")
        self.assertEqual(slugify("   "), "")
        self.assertEqual(slugify(""), "")
        self.assertEqual(slugify("!!!"), "")

    def test_max_length_word_boundary(self):
        text = "Hello World Foo"
        self.assertEqual(slugify(text, max_length=11), "hello-world")
        self.assertEqual(slugify(text, max_length=12), "hello-world")
        self.assertEqual(slugify(text, max_length=14), "hello-world")
        self.assertEqual(slugify(text, max_length=15), "hello-world-foo")
        self.assertEqual(slugify(text, max_length=100), "hello-world-foo")
        self.assertEqual(slugify(text, max_length=5), "hello")
        self.assertEqual(slugify(text, max_length=6), "hello")
        self.assertEqual(slugify(text, max_length=4), "hell")
        self.assertEqual(slugify(text, max_length=None), "hello-world-foo")

    def test_max_length_never_trailing_hyphen(self):
        full = "ab-cd-ef-gh"
        for n in range(1, 15):
            got = slugify("ab cd ef gh", max_length=n)
            self.assertFalse(got.endswith("-"), (n, got))
            self.assertLessEqual(len(got), n)
            self.assertTrue(full.startswith(got), (n, got))
        self.assertEqual(slugify("ab cd ef gh", max_length=3), "ab")
        self.assertEqual(slugify("ab cd ef gh", max_length=1), "a")

    def test_max_length_after_cleanup(self):
        self.assertEqual(slugify("  --Hello--  ", max_length=5), "hello")

    def test_invalid_max_length(self):
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                slugify("hello", max_length=bad)


if __name__ == "__main__":
    unittest.main()
'''

SPEC_REF_MODULE = r'''"""Text helpers for the notes site."""
import re
import unicodedata


def slugify(text, max_length=None):
    """Return a URL slug for text, cut at a word boundary to at most max_length characters."""
    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be at least 1")
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if max_length is None or len(slug) <= max_length:
        return slug
    words = slug.split("-")
    result = ""
    for word in words:
        candidate = word if not result else result + "-" + word
        if len(candidate) > max_length:
            break
        result = candidate
    return result or words[0][:max_length]
'''

SPEC_REF_TESTS = r'''import unittest

from textutil import slugify


class SlugifyTest(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertEqual(slugify("  Cr\u00e8me Br\u00fbl\u00e9e "), "creme-brulee")
        self.assertEqual(slugify("!!!"), "")

    def test_max_length(self):
        self.assertEqual(slugify("hello-world-foo", max_length=12), "hello-world")
        self.assertEqual(slugify("hello", max_length=3), "hel")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            slugify("x", max_length=0)


if __name__ == "__main__":
    unittest.main()
'''

# The model's tests must fail on this stand-in for textutil.py.
SPEC_STUB = "def slugify(text, max_length=None):\n    return text\n"

SPEC_MODULE = Case(
    name="spec_module",
    category="spec_module",
    task=SPEC_TASK,
    check="python3 -m unittest -q test_textutil",
    files={"README.md": SPEC_README},
    hint_files=["README.md"],
    hidden=Hidden({"eval_hidden_spec_module.py": _with_model_tests_guard(
        SPEC_HIDDEN, "test_textutil.py", "", {"textutil.py": SPEC_STUB})},
                  "python3 -m unittest -q eval_hidden_spec_module"),
    reference={"textutil.py": SPEC_REF_MODULE, "test_textutil.py": SPEC_REF_TESTS},
)


# =============================================================================================
# 2. follow_pattern
# =============================================================================================

FP_STORE = r'''"""In-memory storage shared by the handlers."""


class Store:
    def __init__(self):
        self._tables = {}
        self._next_ids = {}

    def all(self, table):
        return [dict(row) for row in self._tables.get(table, {}).values()]

    def get(self, table, item_id):
        row = self._tables.get(table, {}).get(item_id)
        return dict(row) if row is not None else None

    def insert(self, table, record):
        next_id = self._next_ids.get(table, 1)
        self._next_ids[table] = next_id + 1
        row = dict(record, id=next_id)
        self._tables.setdefault(table, {})[next_id] = row
        return dict(row)
'''

FP_APP = r'''"""Tiny framework-free app: a route table mapping (method, path pattern) to handlers."""
import re

from handlers import users

ROUTES = [
    ("GET", r"/users", users.list_users),
    ("GET", r"/users/(?P<user_id>\d+)", users.get_user),
    ("POST", r"/users", users.create_user),
]


def dispatch(store, method, path, body=None):
    """Call the handler for method and path; returns (status, payload)."""
    for route_method, pattern, handler in ROUTES:
        match = re.fullmatch(pattern, path)
        if route_method == method and match:
            kwargs = {name: int(value) for name, value in match.groupdict().items()}
            if method == "POST":
                kwargs["body"] = body if body is not None else {}
            return handler(store, **kwargs)
    return 404, {"error": "no route for %s %s" % (method, path)}
'''

FP_USERS = r'''"""User handlers. Each takes the store (and path params or body) and returns (status, payload)."""
import re

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def list_users(store):
    return 200, {"users": store.all("users")}


def get_user(store, user_id):
    user = store.get("users", user_id)
    if user is None:
        return 404, {"error": "user %d not found" % user_id}
    return 200, user


def create_user(store, body):
    errors = {}
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        errors["name"] = "required"
    email = body.get("email")
    if not isinstance(email, str) or not EMAIL_RE.match(email):
        errors["email"] = "invalid email"
    if errors:
        return 400, {"errors": errors}
    user = store.insert("users", {"name": name.strip(), "email": email})
    return 201, user
'''

FP_TEST_USERS = r'''import unittest

from app import dispatch
from store import Store


class UsersTest(unittest.TestCase):
    def setUp(self):
        self.store = Store()

    def test_create_and_get(self):
        status, user = dispatch(self.store, "POST", "/users", {"name": " Ada ", "email": "ada@example.com"})
        self.assertEqual(status, 201)
        self.assertEqual(user, {"id": 1, "name": "Ada", "email": "ada@example.com"})
        self.assertEqual(dispatch(self.store, "GET", "/users/1"), (200, user))

    def test_get_missing_is_404(self):
        self.assertEqual(dispatch(self.store, "GET", "/users/9"), (404, {"error": "user 9 not found"}))

    def test_list_in_creation_order(self):
        dispatch(self.store, "POST", "/users", {"name": "Ada", "email": "ada@example.com"})
        dispatch(self.store, "POST", "/users", {"name": "Bob", "email": "bob@example.com"})
        status, payload = dispatch(self.store, "GET", "/users")
        self.assertEqual(status, 200)
        self.assertEqual([u["name"] for u in payload["users"]], ["Ada", "Bob"])

    def test_create_validates_each_field(self):
        status, payload = dispatch(self.store, "POST", "/users", {"name": "", "email": "nope"})
        self.assertEqual(status, 400)
        self.assertEqual(payload, {"errors": {"name": "required", "email": "invalid email"}})
        self.assertEqual(dispatch(self.store, "GET", "/users"), (200, {"users": []}))
'''

FP_TASK = """Add orders to this app, following the exact pattern the users feature uses (app.py route table, handlers/users.py, tests/test_users.py).

1. Create handlers/orders.py with list_orders(store), get_order(store, order_id) and create_order(store, body). Like the user handlers, each returns a (status, payload) tuple and keeps data in the "orders" table of the Store.
2. Register three routes in ROUTES in app.py: GET /orders, GET /orders/<order_id> (digits, passed as order_id) and POST /orders.
3. Behaviour:
   - list_orders returns (200, {"orders": [...]}) with every order in creation order.
   - get_order returns (200, order), or (404, {"error": "order <id> not found"}) when there is no such order, for example "order 7 not found".
   - create_order validates the body and returns (400, {"errors": {...}}) with one entry per bad field and no entry for good fields:
     - "user_id" must be an int (a bool does not count) that is the id of an existing user; otherwise the error is "unknown user".
     - "items" must be a non-empty list in which every item is a dict with "sku" (a non-empty string) and "quantity" (an int, not a bool, of at least 1); otherwise the error is "invalid items".
     On success it stores {"user_id": ..., "items": [...], "total_quantity": <sum of the quantities>, "status": "pending"} and returns (201, order), where order is the stored record including its "id". Invalid requests store nothing.
4. Add tests/test_orders.py covering the above, in the same style as tests/test_users.py.

Do not change the users feature or store.py. Done means the check command passes."""

FP_HIDDEN = r'''import hashlib
import unittest

from app import ROUTES, dispatch
from handlers import orders
from store import Store

ITEMS = [{"sku": "A1", "quantity": 2}, {"sku": "B2", "quantity": 3}]
UNCHANGED_SHA256 = @UNCHANGED@


class OrdersHidden(unittest.TestCase):
    def test_users_feature_and_store_unchanged(self):
        for path, expected in sorted(UNCHANGED_SHA256.items()):
            with open(path, "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), expected, path)

    def setUp(self):
        self.store = Store()
        status, _ = dispatch(self.store, "POST", "/users", {"name": "Ada", "email": "ada@example.com"})
        self.assertEqual(status, 201)

    def create(self, body):
        return dispatch(self.store, "POST", "/orders", body)

    def test_routes_registered(self):
        handlers = [(method, handler) for method, _, handler in ROUTES]
        self.assertIn(("GET", orders.list_orders), handlers)
        self.assertIn(("GET", orders.get_order), handlers)
        self.assertIn(("POST", orders.create_order), handlers)

    def test_empty_list(self):
        self.assertEqual(dispatch(self.store, "GET", "/orders"), (200, {"orders": []}))

    def test_create_get_list(self):
        status, order = self.create({"user_id": 1, "items": ITEMS})
        self.assertEqual(status, 201)
        self.assertEqual(order, {"id": 1, "user_id": 1, "items": ITEMS, "total_quantity": 5,
                                 "status": "pending"})
        self.assertEqual(dispatch(self.store, "GET", "/orders/1"), (200, order))
        self.assertEqual(dispatch(self.store, "GET", "/orders"), (200, {"orders": [order]}))

    def test_ids_and_creation_order(self):
        _, first = self.create({"user_id": 1, "items": [{"sku": "A", "quantity": 1}]})
        _, second = self.create({"user_id": 1, "items": [{"sku": "B", "quantity": 4}]})
        self.assertEqual((first["id"], second["id"]), (1, 2))
        self.assertEqual(second["total_quantity"], 4)
        status, payload = dispatch(self.store, "GET", "/orders")
        self.assertEqual(status, 200)
        self.assertEqual([o["id"] for o in payload["orders"]], [1, 2])

    def test_missing_order_is_404(self):
        self.assertEqual(dispatch(self.store, "GET", "/orders/7"), (404, {"error": "order 7 not found"}))

    def test_bad_user_id(self):
        for user_id in (99, "1", True, 1.0, None):
            body = {"items": ITEMS}
            if user_id is not None:
                body["user_id"] = user_id
            with self.subTest(user_id=user_id):
                self.assertEqual(self.create(body), (400, {"errors": {"user_id": "unknown user"}}))

    def test_bad_items(self):
        bad = [None, [], "A1", ["A1"], [{"sku": "", "quantity": 1}], [{"sku": "A", "quantity": 0}],
               [{"sku": "A", "quantity": True}], [{"sku": "A"}], [{"quantity": 1}],
               [{"sku": 5, "quantity": 1}], [{"sku": "A", "quantity": "2"}],
               [{"sku": "A", "quantity": 1}, {"sku": "B", "quantity": -1}]]
        for items in bad:
            body = {"user_id": 1}
            if items is not None:
                body["items"] = items
            with self.subTest(items=items):
                self.assertEqual(self.create(body), (400, {"errors": {"items": "invalid items"}}))

    def test_both_errors_and_nothing_stored(self):
        self.assertEqual(self.create({}), (400, {"errors": {"user_id": "unknown user",
                                                             "items": "invalid items"}}))
        self.assertEqual(dispatch(self.store, "GET", "/orders"), (200, {"orders": []}))


if __name__ == "__main__":
    unittest.main()
'''.replace("@UNCHANGED@", repr({"store.py": _sha256(FP_STORE),
                                     "handlers/users.py": _sha256(FP_USERS)}))

FP_REF_ORDERS = r'''"""Order handlers. Each takes the store (and path params or body) and returns (status, payload)."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_item(item):
    return (isinstance(item, dict) and isinstance(item.get("sku"), str) and item["sku"] != ""
            and _is_int(item.get("quantity")) and item["quantity"] >= 1)


def list_orders(store):
    return 200, {"orders": store.all("orders")}


def get_order(store, order_id):
    order = store.get("orders", order_id)
    if order is None:
        return 404, {"error": "order %d not found" % order_id}
    return 200, order


def create_order(store, body):
    errors = {}
    user_id = body.get("user_id")
    if not _is_int(user_id) or store.get("users", user_id) is None:
        errors["user_id"] = "unknown user"
    items = body.get("items")
    if not isinstance(items, list) or not items or not all(_valid_item(i) for i in items):
        errors["items"] = "invalid items"
    if errors:
        return 400, {"errors": errors}
    clean = [{"sku": i["sku"], "quantity": i["quantity"]} for i in items]
    order = store.insert("orders", {"user_id": user_id, "items": clean,
                                    "total_quantity": sum(i["quantity"] for i in clean),
                                    "status": "pending"})
    return 201, order
'''

FP_REF_APP = _replace_once(
    '''from handlers import users

ROUTES = [
    ("GET", r"/users", users.list_users),
    ("GET", r"/users/(?P<user_id>\\d+)", users.get_user),
    ("POST", r"/users", users.create_user),
]''',
    '''from handlers import orders, users

ROUTES = [
    ("GET", r"/users", users.list_users),
    ("GET", r"/users/(?P<user_id>\\d+)", users.get_user),
    ("POST", r"/users", users.create_user),
    ("GET", r"/orders", orders.list_orders),
    ("GET", r"/orders/(?P<order_id>\\d+)", orders.get_order),
    ("POST", r"/orders", orders.create_order),
]''')

FP_REF_TESTS = r'''import unittest

from app import dispatch
from store import Store


class OrdersTest(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        dispatch(self.store, "POST", "/users", {"name": "Ada", "email": "ada@example.com"})

    def test_create_and_get(self):
        status, order = dispatch(self.store, "POST", "/orders",
                                 {"user_id": 1, "items": [{"sku": "A", "quantity": 2}]})
        self.assertEqual(status, 201)
        self.assertEqual(order["total_quantity"], 2)
        self.assertEqual(dispatch(self.store, "GET", "/orders/1"), (200, order))
        self.assertEqual(dispatch(self.store, "GET", "/orders"), (200, {"orders": [order]}))

    def test_missing_is_404(self):
        self.assertEqual(dispatch(self.store, "GET", "/orders/3"), (404, {"error": "order 3 not found"}))

    def test_validation(self):
        self.assertEqual(dispatch(self.store, "POST", "/orders", {"user_id": 5, "items": []}),
                         (400, {"errors": {"user_id": "unknown user", "items": "invalid items"}}))
'''

# The model's tests must fail when the orders handlers are stand-ins that do nothing.
FP_STUB_ORDERS = (
    "def list_orders(store):\n    return 501, {}\n\n\n"
    "def get_order(store, order_id):\n    return 501, {}\n\n\n"
    "def create_order(store, body):\n    return 501, {}\n"
)

FOLLOW_PATTERN = Case(
    name="follow_pattern",
    category="follow_pattern",
    task=FP_TASK,
    check=UNITTEST_TESTS_DIR,
    files={
        "app.py": FP_APP,
        "store.py": FP_STORE,
        "handlers/__init__.py": "",
        "handlers/users.py": FP_USERS,
        "tests/__init__.py": "",
        "tests/test_users.py": FP_TEST_USERS,
    },
    hint_files=["app.py", "handlers/users.py", "tests/test_users.py"],
    hidden=Hidden({"eval_hidden_follow_pattern.py": _with_model_tests_guard(
        FP_HIDDEN, "tests/test_orders.py", "", {"handlers/orders.py": FP_STUB_ORDERS})},
                  "python3 -m unittest -q eval_hidden_follow_pattern tests.test_users"),
    reference={"handlers/orders.py": FP_REF_ORDERS, "app.py": FP_REF_APP,
               "tests/test_orders.py": FP_REF_TESTS},
)


# =============================================================================================
# 3. rename_across_files
# =============================================================================================

RN_DB = r'''"""Tiny in-memory user table."""

_USERS = {
    1: {"name": "Ada Lovelace", "email": "ada@example.com", "plan": "pro", "active": True},
    2: {"name": "Grace Hopper", "email": "grace@example.com", "plan": "free", "active": True},
    3: {"name": "Alan Turing", "email": "alan@example.com", "plan": "pro", "active": False},
}


def get_usr(usr_id):
    """Return a copy of the user record for usr_id (with its id), or None if there is none."""
    record = _USERS.get(usr_id)
    return dict(record, id=usr_id) if record else None


def all_ids():
    return sorted(_USERS)
'''

RN_PROFILES = r'''"""Profile views built from user records."""
from crm.db import get_usr


def display_name(usr_id):
    """Return "Name <email>" for usr_id, or "unknown user" if there is none."""
    record = get_usr(usr_id)
    if record is None:
        return "unknown user"
    return "%s <%s>" % (record["name"], record["email"])


def profile_card(usr_id):
    record = get_usr(usr_id=usr_id)
    if record is None:
        return None
    return {"id": usr_id, "title": record["name"], "plan": record["plan"].upper(),
            "status": "active" if record["active"] else "inactive"}
'''

RN_BILLING = r'''"""Monthly pricing per user."""
from crm.db import get_usr

PRICES = {"free": 0, "pro": 1200}


def monthly_price(usr_id):
    record = get_usr(usr_id=usr_id)
    if record is None:
        raise KeyError(usr_id)
    return PRICES[record["plan"]]


def invoice_line(usr_id, months=1):
    # usr_id must exist: monthly_price raises KeyError otherwise
    cents = monthly_price(usr_id) * months
    return "%s: %d month(s), $%d.%02d" % (get_usr(usr_id)["name"], months, cents // 100, cents % 100)
'''

RN_NOTIFICATIONS = r'''"""Outgoing email payloads."""
from crm import db


def build_email(usr_id, subject, body):
    """Return an email dict for an active user, or None for an unknown or inactive one."""
    record = db.get_usr(usr_id)
    if record is None or not record["active"]:
        return None
    return {"to": record["email"], "subject": subject, "body": body}
'''

RN_REPORTS = r'''"""Aggregate numbers for the admin page."""
from crm import db


def active_count():
    return sum(1 for uid in db.all_ids() if db.get_usr(uid)["active"])


def plan_summary():
    summary = {}
    for uid in db.all_ids():
        plan = db.get_usr(usr_id=uid)["plan"]
        summary[plan] = summary.get(plan, 0) + 1
    return summary
'''

RN_API = r'''"""Request handlers for the admin API."""
from crm.billing import monthly_price
from crm.db import get_usr as fetch_user
from crm.profiles import profile_card


def handle_get_user(usr_id):
    if fetch_user(usr_id) is None:
        return 404, {"error": "no user %d" % usr_id}
    card = profile_card(usr_id)
    card["price"] = monthly_price(usr_id=usr_id)
    return 200, card
'''

RN_TEST_DB = r'''import unittest

from crm.db import all_ids, get_usr


class DbTest(unittest.TestCase):
    def test_found(self):
        self.assertEqual(get_usr(1)["name"], "Ada Lovelace")
        self.assertEqual(get_usr(usr_id=2)["id"], 2)

    def test_missing(self):
        self.assertIsNone(get_usr(99))

    def test_returns_copy(self):
        get_usr(1)["name"] = "changed"
        self.assertEqual(get_usr(1)["name"], "Ada Lovelace")

    def test_all_ids(self):
        self.assertEqual(all_ids(), [1, 2, 3])
'''

RN_TEST_PROFILES = r'''import unittest

from crm.profiles import display_name, profile_card


class ProfilesTest(unittest.TestCase):
    def test_display_name(self):
        self.assertEqual(display_name(1), "Ada Lovelace <ada@example.com>")
        self.assertEqual(display_name(usr_id=42), "unknown user")

    def test_profile_card(self):
        self.assertEqual(profile_card(usr_id=3), {"id": 3, "title": "Alan Turing", "plan": "PRO",
                                                  "status": "inactive"})
        self.assertIsNone(profile_card(42))
'''

RN_TEST_BILLING = r'''import unittest

from crm.billing import invoice_line, monthly_price


class BillingTest(unittest.TestCase):
    def test_monthly_price(self):
        self.assertEqual(monthly_price(1), 1200)
        self.assertEqual(monthly_price(usr_id=2), 0)
        with self.assertRaises(KeyError):
            monthly_price(99)

    def test_invoice_line(self):
        self.assertEqual(invoice_line(1, months=3), "Ada Lovelace: 3 month(s), $36.00")
'''

RN_TEST_SERVICES = r'''import unittest

from crm import api, notifications, reports


class ServicesTest(unittest.TestCase):
    def test_build_email(self):
        self.assertEqual(notifications.build_email(2, "Hi", "Body"),
                         {"to": "grace@example.com", "subject": "Hi", "body": "Body"})
        self.assertIsNone(notifications.build_email(usr_id=3, subject="Hi", body="Body"))

    def test_reports(self):
        self.assertEqual(reports.active_count(), 2)
        self.assertEqual(reports.plan_summary(), {"pro": 2, "free": 1})

    def test_api(self):
        status, card = api.handle_get_user(usr_id=1)
        self.assertEqual(status, 200)
        self.assertEqual(card["price"], 1200)
        self.assertEqual(api.handle_get_user(99), (404, {"error": "no user 99"}))
'''

RN_TASK = """Rename the function get_usr to get_user in crm/db.py, and its parameter usr_id to user_id. Update every caller and every test to the new names. Also rename the usr_id parameters and variables of the other crm functions to user_id (for example display_name(usr_id) becomes display_name(user_id)), including in comments and docstrings.

Keep backwards compatibility: crm/db.py must still define get_usr(usr_id) as a deprecated alias that calls warnings.warn(..., DeprecationWarning, stacklevel=2) and returns get_user(usr_id). Old callers using get_usr(1) or get_usr(usr_id=1) must keep working.

After the change, that alias function in crm/db.py must be the only place in any .py file where the names get_usr or usr_id appear (comments, docstrings and tests included). Do not add tests for the alias. Do not change any other behaviour.

Done means the check command passes."""

RN_HIDDEN = r'''import ast
import inspect
import os
import re
import unittest
import warnings

ALIAS_FILE = os.path.join("crm", "db.py")
OLD_NAMES = re.compile(r"\b(get_usr|usr_id)\b")


def alias_lines(source):
    """Line numbers of the top-level statement(s) that define get_usr."""
    lines = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_usr":
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "get_usr" for t in node.targets):
            start = node.lineno
        else:
            continue
        lines.update(range(start, node.end_lineno + 1))
    return lines


class RenameHidden(unittest.TestCase):
    def test_no_old_names_outside_alias(self):
        hits = []
        for dirpath, dirnames, filenames in os.walk("."):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
            for name in filenames:
                if not name.endswith(".py") or name.startswith("eval_hidden_"):
                    continue
                path = os.path.normpath(os.path.join(dirpath, name))
                with open(path, encoding="utf-8") as f:
                    source = f.read()
                skip = alias_lines(source) if path == ALIAS_FILE else set()
                for number, line in enumerate(source.splitlines(), 1):
                    if number not in skip and OLD_NAMES.search(line):
                        hits.append("%s:%d: %s" % (path, number, line.strip()))
        self.assertEqual(hits, [])

    def test_get_user_does_not_warn(self):
        from crm.db import get_user
        self.assertIn("user_id", inspect.signature(get_user).parameters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(get_user(user_id=1)["name"], "Ada Lovelace")
            self.assertIsNone(get_user(99))
        self.assertEqual([w for w in caught if issubclass(w.category, DeprecationWarning)], [])

    def test_alias_warns_and_forwards(self):
        from crm.db import get_user, get_usr
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(get_usr(2), get_user(2))
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(get_usr(usr_id=1), get_user(1))
        with self.assertWarns(DeprecationWarning):
            self.assertIsNone(get_usr(99))

    def test_keywords_renamed(self):
        from crm import api, billing, notifications, profiles
        self.assertEqual(profiles.display_name(user_id=1), "Ada Lovelace <ada@example.com>")
        self.assertEqual(profiles.profile_card(user_id=2)["plan"], "FREE")
        self.assertEqual(billing.monthly_price(user_id=1), 1200)
        self.assertEqual(billing.invoice_line(user_id=1, months=2), "Ada Lovelace: 2 month(s), $24.00")
        self.assertIsNone(notifications.build_email(user_id=3, subject="s", body="b"))
        self.assertEqual(api.handle_get_user(user_id=1)[0], 200)

    def test_behaviour_kept(self):
        from crm import api, reports
        self.assertEqual(reports.active_count(), 2)
        self.assertEqual(reports.plan_summary(), {"pro": 2, "free": 1})
        self.assertEqual(api.handle_get_user(99), (404, {"error": "no user 99"}))


if __name__ == "__main__":
    unittest.main()
'''

RN_REF_DB = r'''"""Tiny in-memory user table."""
import warnings

_USERS = {
    1: {"name": "Ada Lovelace", "email": "ada@example.com", "plan": "pro", "active": True},
    2: {"name": "Grace Hopper", "email": "grace@example.com", "plan": "free", "active": True},
    3: {"name": "Alan Turing", "email": "alan@example.com", "plan": "pro", "active": False},
}


def get_user(user_id):
    """Return a copy of the user record for user_id (with its id), or None if there is none."""
    record = _USERS.get(user_id)
    return dict(record, id=user_id) if record else None


def get_usr(usr_id):
    """Deprecated: use get_user."""
    warnings.warn("get_usr is deprecated; use get_user", DeprecationWarning, stacklevel=2)
    return get_user(usr_id)


def all_ids():
    return sorted(_USERS)
'''


def _rename(text: str) -> str:
    return re.sub(r"\bget_usr\b", "get_user", re.sub(r"\busr_id\b", "user_id", text))


RENAME_ACROSS_FILES = Case(
    name="rename_across_files",
    category="rename_across_files",
    task=RN_TASK,
    check=UNITTEST_TESTS_DIR,
    files={
        "crm/__init__.py": "",
        "crm/db.py": RN_DB,
        "crm/profiles.py": RN_PROFILES,
        "crm/billing.py": RN_BILLING,
        "crm/notifications.py": RN_NOTIFICATIONS,
        "crm/reports.py": RN_REPORTS,
        "crm/api.py": RN_API,
        "tests/__init__.py": "",
        "tests/test_db.py": RN_TEST_DB,
        "tests/test_profiles.py": RN_TEST_PROFILES,
        "tests/test_billing.py": RN_TEST_BILLING,
        "tests/test_services.py": RN_TEST_SERVICES,
    },
    hint_files=["crm/db.py"],
    hidden=Hidden({"eval_hidden_rename_across_files.py": RN_HIDDEN},
                  "python3 eval_hidden_rename_across_files.py && " + UNITTEST_TESTS_DIR),
    reference=dict(
        {"crm/db.py": RN_REF_DB},
        **{p: _rename for p in ("crm/profiles.py", "crm/billing.py", "crm/notifications.py",
                                "crm/reports.py", "crm/api.py", "tests/test_db.py",
                                "tests/test_profiles.py", "tests/test_billing.py",
                                "tests/test_services.py")}),
    limits={"max_steps": 60},
)


# =============================================================================================
# 4. write_tests
# =============================================================================================

MONEY = r'''"""Money helpers for the billing service. Amounts are integers in cents."""
import re

_AMOUNT = re.compile(r"(?P<sign>[+-])?\$?(?P<whole>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?")


def parse_money(text):
    """Parse a dollar amount string and return it as integer cents.

    Rules:
    - Surrounding whitespace is ignored.
    - An optional sign ("+" or "-") comes first, then an optional "$": "-$1.50" and "-1.50" are
      both -150. "$-1.50" is invalid.
    - The whole part is required: either plain digits ("1234") or digits grouped in threes with
      commas ("1,234", "12,345,678"). Bad grouping such as "1,23" or "12,34,567" is invalid.
    - An optional fraction follows a "." and has at least one digit: "1.5" is 150, "1.05" is 105.
      ".5" and "1." are invalid.
    - A fraction with more than two digits is rounded to the nearest cent, halves away from zero:
      "0.005" is 1, "0.0049" is 0, "-0.005" is -1, "2.675" is 268.
    - Anything else raises ValueError, including "" and "abc".
    """
    match = _AMOUNT.fullmatch(text.strip())
    if match is None:
        raise ValueError("not a money amount: %r" % (text,))
    whole = int(match.group("whole").replace(",", ""))
    frac = match.group("frac") or ""
    cents = whole * 100 + int((frac + "00")[:2])
    if frac[2:3] >= "5":
        cents += 1
    return -cents if match.group("sign") == "-" else cents


def format_money(cents):
    """Format integer cents as a dollar string.

    - Always a "$", a whole part with commas between thousands, and exactly two decimals:
      0 is "$0.00", 5 is "$0.05", 123456 is "$1,234.56", 100000000 is "$1,000,000.00".
    - Negative amounts put the "-" before the "$": -150 is "-$1.50".
    - cents must be an int; anything else (bool and float included) raises TypeError.
    """
    if not isinstance(cents, int) or isinstance(cents, bool):
        raise TypeError("cents must be an int, got %s" % type(cents).__name__)
    sign = "-" if cents < 0 else ""
    whole, rest = divmod(abs(cents), 100)
    return "%s$%s.%02d" % (sign, "{:,}".format(whole), rest)
'''

# Each mutant breaks one documented rule; thorough tests must fail on every one.
MONEY_MUTANTS = {
    "halves_round_down": MONEY.replace('if frac[2:3] >= "5":', 'if frac[2:3] > "5":'),
    "sign_after_dollar": MONEY.replace('return "%s$%s.%02d" % (sign,', 'return "$%s%s.%02d" % (sign,'),
    "loose_grouping": MONEY.replace(r"(?:,\d{3})+", r"(?:,\d{1,3})+"),
}

WT_TASK = """Write thorough unittest tests for money.py in a new file test_money.py in the repo root, next to money.py. Cover everything the docstrings of parse_money and format_money promise: signs, the "$" prefix, whitespace, thousands separators (valid and invalid grouping), rounding of extra fraction digits (halves and negative halves included), invalid input, formatting of zero, small, large and negative amounts, and format_money's type checks.

Do not change money.py. If you believe money.py has a bug, say so in your finish summary instead of fixing it. Done means the check command passes."""

WT_HIDDEN = r'''import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ORIGINAL_SHA256 = @SHA@
MUTANTS = @MUTANTS@


def run_model_tests(cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "unittest", "-q", "test_money"], cwd=cwd, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=60)


class WriteTestsHidden(unittest.TestCase):
    def test_money_unchanged(self):
        with open("money.py", "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), ORIGINAL_SHA256)

    def test_tests_pass_on_original(self):
        result = run_model_tests(".")
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])

    def test_every_mutant_is_caught(self):
        for name, source in sorted(MUTANTS.items()):
            with self.subTest(mutant=name):
                tmp = tempfile.mkdtemp()
                try:
                    copy = os.path.join(tmp, "repo")
                    shutil.copytree(".", copy, ignore=shutil.ignore_patterns(
                        ".git", "__pycache__", "*.pyc", "eval_hidden_*"))
                    with open(os.path.join(copy, "money.py"), "w", encoding="utf-8") as f:
                        f.write(source)
                    result = run_model_tests(copy)
                    self.assertNotEqual(result.returncode, 0, "mutant %s survived the tests" % name)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
'''.replace("@SHA@", repr(_sha256(MONEY))).replace("@MUTANTS@", repr(MONEY_MUTANTS))

WT_REF_TESTS = r'''import unittest

from money import format_money, parse_money


class ParseMoneyTest(unittest.TestCase):
    def test_plain_amounts(self):
        cases = {"0": 0, "1": 100, "1.5": 150, "1.05": 105, "1234": 123400, "$12.34": 1234,
                 "+$1.00": 100, "+2": 200, "  $3.10\n": 310}
        for text, cents in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_money(text), cents)

    def test_negative(self):
        self.assertEqual(parse_money("-1.50"), -150)
        self.assertEqual(parse_money("-$1.50"), -150)
        self.assertEqual(parse_money("-$1,234.56"), -123456)

    def test_thousands(self):
        self.assertEqual(parse_money("1,234"), 123400)
        self.assertEqual(parse_money("12,345,678.90"), 1234567890)
        self.assertEqual(parse_money("$1,000.01"), 100001)

    def test_bad_grouping(self):
        for text in ("1,23", "12,34,567", "1234,567", ",123", "1,", "1,2345"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_money(text)

    def test_rounding(self):
        cases = {"0.005": 1, "0.0049": 0, "0.004": 0, "2.675": 268, "1.999": 200, "0.0051": 1}
        for text, cents in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_money(text), cents)

    def test_negative_rounding(self):
        self.assertEqual(parse_money("-0.005"), -1)
        self.assertEqual(parse_money("-2.675"), -268)
        self.assertEqual(parse_money("-0.0049"), 0)

    def test_invalid(self):
        for text in ("", "   ", "abc", ".5", "1.", "$-1.50", "1.2.3", "--1", "$", "1 000", "1e3"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_money(text)


class FormatMoneyTest(unittest.TestCase):
    def test_amounts(self):
        cases = {0: "$0.00", 5: "$0.05", 150: "$1.50", 123456: "$1,234.56",
                 100000000: "$1,000,000.00", 99999: "$999.99"}
        for cents, text in cases.items():
            with self.subTest(cents=cents):
                self.assertEqual(format_money(cents), text)

    def test_negative(self):
        self.assertEqual(format_money(-150), "-$1.50")
        self.assertEqual(format_money(-123456), "-$1,234.56")
        self.assertEqual(format_money(-5), "-$0.05")

    def test_type_errors(self):
        for bad in (1.5, "100", True, None):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    format_money(bad)

    def test_round_trip(self):
        for cents in (0, 5, -5, 123456, -100000000):
            self.assertEqual(parse_money(format_money(cents)), cents)


if __name__ == "__main__":
    unittest.main()
'''

WRITE_TESTS = Case(
    name="write_tests",
    category="write_tests",
    task=WT_TASK,
    check="python3 -m unittest -q test_money",
    files={"money.py": MONEY},
    hint_files=["money.py"],
    hidden=Hidden({"eval_hidden_write_tests.py": WT_HIDDEN}, "python3 eval_hidden_write_tests.py"),
    reference={"test_money.py": WT_REF_TESTS},
)


# =============================================================================================
# 5. fix_failing_test
# =============================================================================================

INVENTORY = r'''"""Stock levels and reservations for the shop's checkout."""


class OutOfStock(Exception):
    """Raised when a reservation asks for more units than are available."""


class Inventory:
    def __init__(self, stock=None):
        self.stock = dict(stock or {})
        self.reserved = {}

    def add_stock(self, sku, qty):
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.stock[sku] = self.stock.get(sku, 0) + qty

    def available(self, sku):
        """Units that can still be reserved: stock minus what is already reserved."""
        return self.stock.get(sku, 0) - self.reserved.get(sku, 0)

    def reserve(self, sku, qty):
        """Hold qty units for an order. Raises OutOfStock if fewer than qty are available."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        if self.stock.get(sku, 0) < qty:
            raise OutOfStock(sku)
        self.reserved[sku] = self.reserved.get(sku, 0) + qty

    def release(self, sku, qty):
        """Give back qty reserved units (an order was cancelled)."""
        held = self.reserved.get(sku, 0)
        if qty <= 0 or qty > held:
            raise ValueError("cannot release %d of %s" % (qty, sku))
        self.reserved[sku] = held - qty

    def ship(self, sku, qty):
        """Ship qty reserved units: they leave both the stock and the reservations."""
        held = self.reserved.get(sku, 0)
        if qty <= 0 or qty > held:
            raise ValueError("cannot ship %d of %s" % (qty, sku))
        self.reserved[sku] = held - qty
        self.stock[sku] -= qty
'''

INVENTORY_TESTS = r'''import unittest

from inventory import Inventory, OutOfStock


class InventoryTest(unittest.TestCase):
    def test_reserve_reduces_available(self):
        inv = Inventory({"mug": 5})
        inv.reserve("mug", 2)
        self.assertEqual(inv.available("mug"), 3)

    def test_release_and_ship(self):
        inv = Inventory({"mug": 5})
        inv.reserve("mug", 4)
        inv.release("mug", 1)
        inv.ship("mug", 3)
        self.assertEqual(inv.stock["mug"], 2)
        self.assertEqual(inv.available("mug"), 2)

    def test_rejects_non_positive(self):
        inv = Inventory({"mug": 5})
        for qty in (0, -1):
            with self.assertRaises(ValueError):
                inv.reserve("mug", qty)

    def test_unknown_sku_is_out_of_stock(self):
        with self.assertRaises(OutOfStock):
            Inventory().reserve("hat", 1)

    def test_reservations_cannot_exceed_stock(self):
        inv = Inventory({"mug": 5})
        inv.reserve("mug", 3)
        with self.assertRaises(OutOfStock):
            inv.reserve("mug", 3)
        self.assertEqual(inv.available("mug"), 2)
'''

FF_TASK = """One test fails in tests/test_inventory.py (test_reservations_cannot_exceed_stock). Fix the bug in inventory.py so that every test passes. Do not edit, rename or delete any test. Keep the fix small. Done means the check command passes."""

FF_HIDDEN = r'''import hashlib
import unittest

from inventory import Inventory, OutOfStock

TESTS_SHA256 = @SHA@


class FixHidden(unittest.TestCase):
    def test_test_file_unchanged(self):
        with open("tests/test_inventory.py", "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), TESTS_SHA256)

    def test_reserve_exactly_available(self):
        inv = Inventory({"mug": 5})
        inv.reserve("mug", 2)
        inv.reserve("mug", 3)
        self.assertEqual(inv.available("mug"), 0)
        with self.assertRaises(OutOfStock):
            inv.reserve("mug", 1)

    def test_failed_reserve_changes_nothing(self):
        inv = Inventory({"mug": 4})
        inv.reserve("mug", 3)
        with self.assertRaises(OutOfStock):
            inv.reserve("mug", 2)
        self.assertEqual(inv.reserved, {"mug": 3})
        self.assertEqual(inv.available("mug"), 1)

    def test_release_frees_units(self):
        inv = Inventory({"mug": 3})
        inv.reserve("mug", 3)
        inv.release("mug", 2)
        inv.reserve("mug", 2)
        self.assertEqual(inv.available("mug"), 0)

    def test_after_ship(self):
        inv = Inventory({"mug": 5})
        inv.reserve("mug", 3)
        inv.ship("mug", 3)
        inv.reserve("mug", 2)
        with self.assertRaises(OutOfStock):
            inv.reserve("mug", 1)

    def test_add_stock_and_skus_independent(self):
        inv = Inventory({"mug": 1, "hat": 2})
        inv.reserve("mug", 1)
        inv.reserve("hat", 2)
        with self.assertRaises(OutOfStock):
            inv.reserve("mug", 1)
        inv.add_stock("mug", 2)
        inv.reserve("mug", 2)
        self.assertEqual((inv.available("mug"), inv.available("hat")), (0, 0))


if __name__ == "__main__":
    unittest.main()
'''.replace("@SHA@", repr(_sha256(INVENTORY_TESTS)))

FIX_FAILING_TEST = Case(
    name="fix_failing_test",
    category="fix_failing_test",
    task=FF_TASK,
    check=UNITTEST_TESTS_DIR,
    files={"inventory.py": INVENTORY, "tests/__init__.py": "", "tests/test_inventory.py": INVENTORY_TESTS},
    hint_files=["tests/test_inventory.py"],
    hidden=Hidden({"eval_hidden_fix_failing_test.py": FF_HIDDEN},
                  "python3 -m unittest -q eval_hidden_fix_failing_test tests.test_inventory"),
    reference={"inventory.py": _replace_once("if self.stock.get(sku, 0) < qty:",
                                             "if self.available(sku) < qty:")},
)


# =============================================================================================
# 6. large_file_edit
# =============================================================================================

REPORT = r'''"""Sales reporting helpers: parsing, statistics, formatting, export, grouping and reports.

Rows are dicts with "date" (datetime.date), "region" (str), "product" (str) and "amount"
(int, cents). Every function here is pure and uses only the standard library.
"""
import csv
import io
import math
from collections import OrderedDict
from datetime import date, datetime, timedelta

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "EUR ", "GBP": "GBP "}
DEFAULT_WIDTH = 72
SPARK_CHARS = " .:-=+*#"
REGIONS = ("north", "south", "east", "west")
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%Y%m%d")


class ReportError(ValueError):
    """Raised for rows or arguments the report cannot use."""


# -- parsing ---------------------------------------------------------------------------------


def parse_date(text):
    """Parse a date in one of DATE_FORMATS."""
    text = text.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ReportError("unrecognised date: %r" % text)


def parse_amount(text):
    """Parse "12.34" or "1,234.5" into integer cents."""
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        raise ReportError("empty amount")
    negative = cleaned.startswith("-")
    if negative:
        cleaned = cleaned[1:]
    whole, _, frac = cleaned.partition(".")
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise ReportError("bad amount: %r" % text)
    cents = int(whole) * 100 + int((frac + "00")[:2])
    return -cents if negative else cents


def parse_row(raw):
    """Turn a CSV dict row into a report row."""
    missing = [k for k in ("date", "region", "product", "amount") if not raw.get(k)]
    if missing:
        raise ReportError("row is missing: %s" % ", ".join(missing))
    region = raw["region"].strip().lower()
    if region not in REGIONS:
        raise ReportError("unknown region: %r" % raw["region"])
    return {
        "date": parse_date(raw["date"]),
        "region": region,
        "product": raw["product"].strip(),
        "amount": parse_amount(raw["amount"]),
    }


def load_csv(text, strict=True):
    """Parse CSV text with a header row. With strict=False, bad rows are skipped and reported."""
    rows, errors = [], []
    reader = csv.DictReader(io.StringIO(text))
    for line_no, raw in enumerate(reader, start=2):
        try:
            rows.append(parse_row(raw))
        except ReportError as exc:
            if strict:
                raise ReportError("line %d: %s" % (line_no, exc))
            errors.append((line_no, str(exc)))
    return rows, errors


# -- statistics ------------------------------------------------------------------------------


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values, pct):
    """Nearest-rank percentile; pct is between 0 and 100."""
    if not 0 <= pct <= 100:
        raise ReportError("pct must be between 0 and 100")
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if pct == 0:
        return float(ordered[0])
    rank = int(math.ceil(pct / 100.0 * len(ordered)))
    return float(ordered[rank - 1])


def stddev(values):
    """Population standard deviation."""
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def moving_average(values, window):
    if window < 1:
        raise ReportError("window must be at least 1")
    values = list(values)
    out = []
    running = 0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        out.append(running / float(min(i + 1, window)))
    return out


def percent_change(old, new):
    """Change from old to new in percent, or None when old is zero."""
    if old == 0:
        return None
    return (new - old) * 100.0 / abs(old)


def safe_ratio(part, whole, default=0.0):
    if not whole:
        return default
    return part / float(whole)


def cumulative(values):
    total = 0
    out = []
    for value in values:
        total += value
        out.append(total)
    return out


def rank(values, reverse=True):
    """Dense rank of each value (1 is the largest by default)."""
    distinct = sorted(set(values), reverse=reverse)
    positions = {v: i + 1 for i, v in enumerate(distinct)}
    return [positions[v] for v in values]


def normalize(values):
    """Scale values to the 0..1 range."""
    values = list(values)
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(v - low) / float(high - low) for v in values]


def outliers(values, threshold=2.0):
    """Values more than threshold standard deviations from the mean."""
    values = list(values)
    avg, dev = mean(values), stddev(values)
    if dev == 0:
        return []
    return [v for v in values if abs(v - avg) / dev > threshold]


# -- formatting ------------------------------------------------------------------------------


def format_amount(cents, currency="USD"):
    symbol = CURRENCY_SYMBOLS.get(currency)
    if symbol is None:
        raise ReportError("unknown currency: %s" % currency)
    sign = "-" if cents < 0 else ""
    whole, rest = divmod(abs(cents), 100)
    return "%s%s%s.%02d" % (sign, symbol, "{:,}".format(whole), rest)


def format_percent(value, digits=1):
    if value is None:
        return "n/a"
    return "%+.*f%%" % (digits, value)


def format_date(day, style="iso"):
    if style == "iso":
        return day.isoformat()
    if style == "short":
        return day.strftime("%d %b")
    if style == "long":
        return day.strftime("%A %d %B %Y")
    raise ReportError("unknown date style: %s" % style)


def pad(text, width, align="left"):
    text = str(text)
    if align == "left":
        return text.ljust(width)
    if align == "right":
        return text.rjust(width)
    if align == "center":
        return text.center(width)
    raise ReportError("unknown alignment: %s" % align)


def truncate(text, width, marker="..."):
    text = str(text)
    if len(text) <= width:
        return text
    if width <= len(marker):
        return marker[:width]
    return text[: width - len(marker)] + marker


def column_widths(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    return widths


def render_table(headers, rows, align=None, max_width=DEFAULT_WIDTH):
    """Render rows as a plain-text table. align is a list of "left"/"right" per column."""
    align = align or ["left"] * len(headers)
    widths = column_widths(headers, rows)
    sep = "  "
    lines = [sep.join(pad(h, w, a) for h, w, a in zip(headers, widths, align)).rstrip(),
             sep.join("-" * w for w in widths)]
    for row in rows:
        lines.append(sep.join(pad(c, w, a) for c, w, a in zip(row, widths, align)).rstrip())
    return "\n".join(truncate(line, max_width) for line in lines)


def render_bar(value, max_value, width=20, char="#"):
    if max_value <= 0 or value <= 0:
        return ""
    filled = int(round(width * min(value, max_value) / float(max_value)))
    return char * filled


def render_sparkline(values):
    values = list(values)
    if not values:
        return ""
    low, high = min(values), max(values)
    span = high - low
    if span == 0:
        return SPARK_CHARS[len(SPARK_CHARS) // 2] * len(values)
    steps = len(SPARK_CHARS) - 1
    return "".join(SPARK_CHARS[int(round((v - low) * steps / float(span)))] for v in values)


def indent(text, spaces=2):
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


# -- export ----------------------------------------------------------------------------------


def to_csv(headers, rows):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def _md_escape(cell):
    return str(cell).replace("|", "\\|")


def to_markdown(headers, rows):
    lines = ["| " + " | ".join(_md_escape(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
    return "\n".join(lines)


def rows_to_records(headers, rows):
    return [OrderedDict(zip(headers, row)) for row in rows]


# -- grouping --------------------------------------------------------------------------------


def group_by(rows, key):
    """Group rows by row[key] (or key(row) if key is callable), keeping first-seen order."""
    groups = OrderedDict()
    for row in rows:
        k = key(row) if callable(key) else row[key]
        groups.setdefault(k, []).append(row)
    return groups


def sum_by(rows, key, field="amount"):
    totals = OrderedDict()
    for k, items in group_by(rows, key).items():
        totals[k] = sum(item[field] for item in items)
    return totals


def count_by(rows, key):
    counts = OrderedDict()
    for k, items in group_by(rows, key).items():
        counts[k] = len(items)
    return counts


def top_n(counts, n=5):
    """Return the n largest (key, count) pairs of counts, largest first."""
    if n <= 0:
        return []
    items = list(counts.items())
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:n]


def share_by(rows, key):
    """Each group's share of the total amount, as a fraction."""
    totals = sum_by(rows, key)
    grand = sum(totals.values())
    return OrderedDict((k, safe_ratio(v, grand)) for k, v in totals.items())


def pivot(rows, row_key, col_key, field="amount"):
    """Nested totals: {row value: {column value: total}}."""
    table = OrderedDict()
    for row in rows:
        inner = table.setdefault(row[row_key], OrderedDict())
        inner[row[col_key]] = inner.get(row[col_key], 0) + row[field]
    return table


def pivot_columns(table):
    columns = []
    for inner in table.values():
        for col in inner:
            if col not in columns:
                columns.append(col)
    return columns


# -- periods ---------------------------------------------------------------------------------


def week_start(day):
    """Monday of the week that contains day."""
    return day - timedelta(days=day.weekday())


def month_start(day):
    return date(day.year, day.month, 1)


def quarter_of(day):
    return (day.month - 1) // 3 + 1


def date_range(start, end):
    """Every date from start to end, inclusive."""
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def fill_missing_days(totals, start, end, fill=0):
    """Return an OrderedDict with an entry for every day from start to end."""
    return OrderedDict((day, totals.get(day, fill)) for day in date_range(start, end))


def bucket_by_week(rows):
    return sum_by(sorted(rows, key=lambda r: r["date"]), lambda r: week_start(r["date"]))


def bucket_by_month(rows):
    return sum_by(sorted(rows, key=lambda r: r["date"]), lambda r: month_start(r["date"]))


def bucket_by_quarter(rows):
    return sum_by(sorted(rows, key=lambda r: r["date"]),
                  lambda r: "%d-Q%d" % (r["date"].year, quarter_of(r["date"])))


def parse_period(text):
    """Parse "2024-Q2" or "2024-05" into a (start, end) pair of dates, end inclusive."""
    text = text.strip().upper()
    year_text, _, part = text.partition("-")
    if not year_text.isdigit() or not part:
        raise ReportError("bad period: %r" % text)
    year = int(year_text)
    if part.startswith("Q") and part[1:].isdigit() and 1 <= int(part[1:]) <= 4:
        first_month = (int(part[1:]) - 1) * 3 + 1
        start = date(year, first_month, 1)
        end_month = first_month + 2
    elif part.isdigit() and 1 <= int(part) <= 12:
        start = date(year, int(part), 1)
        end_month = int(part)
    else:
        raise ReportError("bad period: %r" % text)
    if end_month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, end_month + 1, 1) - timedelta(days=1)
    return start, end


def rows_in_period(rows, period):
    start, end = parse_period(period)
    return [r for r in rows if start <= r["date"] <= end]


def compare_periods(rows, first, second):
    """Revenue per region in two periods and the change between them."""
    a = sum_by(rows_in_period(rows, first), "region")
    b = sum_by(rows_in_period(rows, second), "region")
    out = OrderedDict()
    for region in REGIONS:
        old, new = a.get(region, 0), b.get(region, 0)
        out[region] = {"first": old, "second": new, "change": percent_change(old, new)}
    return out


# -- report objects --------------------------------------------------------------------------


class ReportSection:
    """A titled block of text inside a report."""

    def __init__(self, title, body="", notes=None):
        self.title = title
        self.body = body
        self.notes = list(notes or [])

    def add_note(self, note):
        self.notes.append(note)
        return self

    def render(self, width=DEFAULT_WIDTH):
        lines = [self.title, "=" * min(len(self.title), width)]
        if self.body:
            lines.append(self.body)
        for note in self.notes:
            lines.append("* " + truncate(note, width - 2))
        return "\n".join(lines)

    def to_dict(self):
        return {"title": self.title, "body": self.body, "notes": list(self.notes)}


class Report:
    """An ordered list of sections with a title and a generation date."""

    def __init__(self, title, generated=None):
        self.title = title
        self.generated = generated or date.today()
        self.sections = []

    def add_section(self, section):
        if any(s.title == section.title for s in self.sections):
            raise ReportError("duplicate section: %s" % section.title)
        self.sections.append(section)
        return section

    def section(self, title):
        for s in self.sections:
            if s.title == title:
                return s
        raise KeyError(title)

    def render(self, width=DEFAULT_WIDTH):
        parts = ["%s (%s)" % (self.title, format_date(self.generated)), ""]
        for s in self.sections:
            parts.append(s.render(width))
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def to_dict(self):
        return {
            "title": self.title,
            "generated": self.generated.isoformat(),
            "sections": [s.to_dict() for s in self.sections],
        }


# -- sections --------------------------------------------------------------------------------


def region_breakdown(rows, currency="USD"):
    totals = sum_by(rows, "region")
    grand = sum(totals.values())
    largest = max(totals.values()) if totals else 0
    table = []
    for region in REGIONS:
        amount = totals.get(region, 0)
        table.append([region, format_amount(amount, currency),
                      "%.1f%%" % (safe_ratio(amount, grand) * 100), render_bar(amount, largest)])
    body = render_table(["region", "revenue", "share", ""], table,
                        align=["left", "right", "right", "left"])
    return ReportSection("Revenue by region", body)


def product_leaders(rows, n=5):
    counts = count_by(rows, "product")
    leaders = top_n(counts, n)
    body = render_table(["product", "orders"], [[p, c] for p, c in leaders], align=["left", "right"])
    section = ReportSection("Top products", body)
    if len(counts) > n:
        section.add_note("%d other products not shown" % (len(counts) - n))
    return section


def region_products(rows, currency="USD"):
    return ReportSection("Products by region", render_pivot(pivot(rows, "product", "region"), currency))


def render_pivot(table, currency="USD"):
    columns = pivot_columns(table)
    lines = [[key] + [format_amount(inner.get(c, 0), currency) for c in columns]
             for key, inner in table.items()]
    return render_table([""] + list(columns), lines, align=["left"] + ["right"] * len(columns))


def weekly_trend(rows, currency="USD"):
    weekly = bucket_by_week(rows)
    if not weekly:
        return ReportSection("Weekly trend", "no data")
    lines = []
    previous = None
    for start, amount in weekly.items():
        change = format_percent(percent_change(previous, amount)) if previous is not None else ""
        lines.append([format_date(start, "short"), format_amount(amount, currency), change])
        previous = amount
    body = render_table(["week", "revenue", "change"], lines, align=["left", "right", "right"])
    section = ReportSection("Weekly trend", body)
    section.add_note("trend " + render_sparkline(list(weekly.values())))
    return section


def daily_stats(rows, currency="USD"):
    values = list(sum_by(rows, "date").values())
    stats = [
        ["days", str(len(values))],
        ["mean", format_amount(int(round(mean(values))), currency)],
        ["median", format_amount(int(round(median(values))), currency)],
        ["p90", format_amount(int(percentile(values, 90)), currency)],
        ["stddev", format_amount(int(round(stddev(values))), currency)],
    ]
    section = ReportSection("Daily statistics", render_table(["stat", "value"], stats,
                                                             align=["left", "right"]))
    for value in outliers(values):
        section.add_note("outlier day: %s" % format_amount(value, currency))
    return section


def build_sales_report(rows, title="Sales report", generated=None, currency="USD"):
    report = Report(title, generated)
    if not rows:
        report.add_section(ReportSection("Summary", "no sales in this period"))
        return report
    first = min(r["date"] for r in rows)
    last = max(r["date"] for r in rows)
    total = sum(r["amount"] for r in rows)
    report.add_section(ReportSection("Summary", "%d sales from %s to %s, %s in total" % (
        len(rows), format_date(first), format_date(last), format_amount(total, currency))))
    report.add_section(region_breakdown(rows, currency))
    report.add_section(product_leaders(rows))
    report.add_section(region_products(rows, currency))
    report.add_section(weekly_trend(rows, currency))
    report.add_section(daily_stats(rows, currency))
    return report


def report_from_csv(text, **kwargs):
    rows, errors = load_csv(text, strict=False)
    report = build_sales_report(rows, **kwargs)
    if errors:
        skipped = ReportSection("Skipped rows")
        for line_no, message in errors:
            skipped.add_note("line %d: %s" % (line_no, message))
        report.add_section(skipped)
    return report


class ReportBuilder:
    """Fluent helper that collects rows and options, then builds a Report."""

    def __init__(self, title="Sales report"):
        self.title = title
        self.rows = []
        self.currency = "USD"
        self.period = None
        self.generated = None

    def with_rows(self, rows):
        self.rows.extend(rows)
        return self

    def with_currency(self, currency):
        if currency not in CURRENCY_SYMBOLS:
            raise ReportError("unknown currency: %s" % currency)
        self.currency = currency
        return self

    def for_period(self, period):
        parse_period(period)
        self.period = period
        return self

    def generated_on(self, day):
        self.generated = day
        return self

    def selected_rows(self):
        if self.period is None:
            return list(self.rows)
        return rows_in_period(self.rows, self.period)

    def build(self):
        title = self.title if self.period is None else "%s, %s" % (self.title, self.period)
        return build_sales_report(self.selected_rows(), title=title, generated=self.generated,
                                  currency=self.currency)
'''

REPORT_TESTS = r'''import unittest
from datetime import date

import report

CSV = (
    "date,region,product,amount\n"
    "2024-03-04,north,mug,12.50\n"
    "2024-03-05,south,hat,30.00\n"
    "2024-03-05,north,mug,12.50\n"
    "2024-03-12,east,scarf,45.25\n"
    "2024-03-13,west,mug,12.50\n"
    "bad-date,west,mug,1.00\n"
)


class ReportTest(unittest.TestCase):
    def test_parse_amount(self):
        self.assertEqual(report.parse_amount("1,234.5"), 123450)
        self.assertEqual(report.parse_amount("-0.05"), -5)

    def test_percentile(self):
        self.assertEqual(report.percentile([1, 2, 3, 4], 50), 2.0)

    def test_top_n_largest_first(self):
        self.assertEqual(report.top_n({"a": 1, "b": 3, "c": 2}, 2), [("b", 3), ("c", 2)])

    def test_top_n_non_positive_n(self):
        self.assertEqual(report.top_n({"a": 1}, 0), [])

    def test_format_amount(self):
        self.assertEqual(report.format_amount(-123456), "-$1,234.56")

    def test_parse_period(self):
        self.assertEqual(report.parse_period("2024-Q1"), (date(2024, 1, 1), date(2024, 3, 31)))
        self.assertEqual(report.parse_period("2024-02"), (date(2024, 2, 1), date(2024, 2, 29)))

    def test_report_from_csv(self):
        text = report.report_from_csv(CSV, generated=date(2024, 3, 31)).render()
        self.assertIn("Revenue by region", text)
        self.assertIn("Top products", text)
        self.assertIn("line 7", text)
        self.assertIn("$112.75 in total", text)
'''

LF_TASK = """In report.py (about 600 lines), change only the function top_n:
1. When two keys have the same count, order them by key ascending (today they keep the mapping's insertion order).
2. Add a keyword parameter min_count with default 0, so the signature becomes top_n(counts, n=5, min_count=0). Pairs whose count is below min_count are left out before the top n are picked.
Callers that do not pass min_count must get the same results as before, apart from the new tie order. Update top_n's docstring to describe both.

Do not change any other function, class, constant or existing import in report.py (adding an import is fine), and do not reformat the file. Use grep -n to find top_n rather than reading the whole file. Add tests for the new behaviour to tests/test_report.py.

Done means the check command passes."""

LF_HIDDEN = r'''import ast
import inspect
import unittest

import report

ORIGINAL = @ORIGINAL@


def other_nodes(source):
    return [ast.dump(node) for node in ast.parse(source).body
            if not (isinstance(node, ast.FunctionDef) and node.name == "top_n")]


class LargeFileHidden(unittest.TestCase):
    def test_tie_order(self):
        counts = {"b": 2, "a": 2, "c": 5, "d": 1}
        self.assertEqual(report.top_n(counts, 3), [("c", 5), ("a", 2), ("b", 2)])
        self.assertEqual(report.top_n({"z": 1, "y": 1, "x": 1}), [("x", 1), ("y", 1), ("z", 1)])

    def test_min_count(self):
        counts = {"b": 2, "a": 2, "c": 5, "d": 1}
        self.assertEqual(report.top_n(counts, n=10, min_count=2), [("c", 5), ("a", 2), ("b", 2)])
        self.assertEqual(report.top_n(counts, min_count=6), [])

    def test_min_count_applies_before_n(self):
        self.assertEqual(report.top_n({"x": 1, "y": 1, "z": 9}, n=2, min_count=2), [("z", 9)])

    def test_defaults_unchanged(self):
        params = inspect.signature(report.top_n).parameters
        self.assertEqual(list(params), ["counts", "n", "min_count"])
        self.assertEqual((params["n"].default, params["min_count"].default), (5, 0))
        self.assertEqual(report.top_n({"a": 0, "b": 0}), [("a", 0), ("b", 0)])
        self.assertEqual(len(report.top_n({k: i for i, k in enumerate("abcdefg")})), 5)
        self.assertEqual(report.top_n({"a": 3}, 0), [])

    def test_everything_else_unchanged(self):
        with open("report.py", encoding="utf-8") as f:
            current = f.read()
        tops = [n for n in ast.parse(current).body if isinstance(n, ast.FunctionDef) and n.name == "top_n"]
        self.assertEqual(len(tops), 1)
        original = other_nodes(ORIGINAL)
        # New imports are allowed; every other top-level statement must match the original, in order.
        now = [d for d in other_nodes(current)
               if d in original or not d.startswith(("Import(", "ImportFrom("))]
        self.assertEqual(now, original)


if __name__ == "__main__":
    unittest.main()
'''.replace("@ORIGINAL@", repr(REPORT))

LF_OLD_TOP_N = '''def top_n(counts, n=5):
    """Return the n largest (key, count) pairs of counts, largest first."""
    if n <= 0:
        return []
    items = list(counts.items())
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:n]
'''

LF_NEW_TOP_N = '''def top_n(counts, n=5, min_count=0):
    """Return the n largest (key, count) pairs of counts, largest first.

    Ties are ordered by key ascending. Pairs whose count is below min_count are left out
    before the top n are picked.
    """
    if n <= 0:
        return []
    items = [item for item in counts.items() if item[1] >= min_count]
    items.sort(key=lambda item: (-item[1], item[0]))
    return items[:n]
'''

LF_REF_TESTS = '''

class TopNTest(unittest.TestCase):
    def test_ties_by_key(self):
        self.assertEqual(report.top_n({"b": 2, "a": 2, "c": 5}, 3), [("c", 5), ("a", 2), ("b", 2)])

    def test_min_count(self):
        self.assertEqual(report.top_n({"b": 2, "a": 1}, min_count=2), [("b", 2)])
'''

LARGE_FILE_EDIT = Case(
    name="large_file_edit",
    category="large_file_edit",
    task=LF_TASK,
    check=UNITTEST_TESTS_DIR,
    files={"report.py": REPORT, "tests/__init__.py": "", "tests/test_report.py": REPORT_TESTS},
    hint_files=["tests/test_report.py"],
    hidden=Hidden({"eval_hidden_large_file_edit.py": _with_model_tests_guard(
        LF_HIDDEN, "tests/test_report.py", REPORT_TESTS, {"report.py": REPORT})},
                  "python3 -m unittest -q eval_hidden_large_file_edit tests.test_report"),
    reference={"report.py": _replace_once(LF_OLD_TOP_N, LF_NEW_TOP_N),
               "tests/test_report.py": lambda text: text + LF_REF_TESTS},
)


# =============================================================================================
# 7. add_types_docs
# =============================================================================================

UTILS = r'''"""Small helpers shared across the service."""
import re
from collections import OrderedDict

_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


def chunk(items, size):
    if size < 1:
        raise ValueError("size must be at least 1")
    return [items[i:i + size] for i in range(0, len(items), size)]


def flatten(nested):
    out = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


def dedupe(items):
    return list(OrderedDict.fromkeys(items))


def clamp(value, low, high):
    if low > high:
        raise ValueError("low must not be greater than high")
    return max(low, min(high, value))


def parse_bool(text):
    lowered = text.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off", ""):
        return False
    raise ValueError("not a boolean: %r" % text)


def _words(name):
    return re.split(r"[\s\-_]+", _CAMEL_BOUNDARY.sub(r"\1 \2", name).strip())


def snake_case(name):
    return "_".join(word.lower() for word in _words(name) if word)


def safe_div(a, b, default=0.0):
    return a / b if b else default


def group_by(items, key):
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups


def first(items, predicate=None, default=None):
    for item in items:
        if predicate is None or predicate(item):
            return item
    return default


def merge_dicts(*dicts, **overrides):
    merged = {}
    for d in dicts:
        merged.update(d)
    merged.update(overrides)
    return merged


def retry(func, attempts=3, exceptions=(Exception,)):
    last = None
    for _ in range(attempts):
        try:
            return func()
        except exceptions as exc:
            last = exc
    raise last
'''

UTILS_TESTS = r'''import unittest

import utils


class UtilsTest(unittest.TestCase):
    def test_chunk(self):
        self.assertEqual(utils.chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
        with self.assertRaises(ValueError):
            utils.chunk([1], 0)

    def test_flatten(self):
        self.assertEqual(utils.flatten([1, [2, (3, [4])], 5]), [1, 2, 3, 4, 5])

    def test_dedupe(self):
        self.assertEqual(utils.dedupe([3, 1, 3, 2, 1]), [3, 1, 2])

    def test_clamp(self):
        self.assertEqual(utils.clamp(15, 0, 10), 10)
        self.assertEqual(utils.clamp(-1, 0, 10), 0)

    def test_parse_bool(self):
        self.assertTrue(utils.parse_bool(" Yes "))
        self.assertFalse(utils.parse_bool(""))
        with self.assertRaises(ValueError):
            utils.parse_bool("maybe")

    def test_snake_case(self):
        self.assertEqual(utils.snake_case("HTTPServerError"), "httpserver_error")
        self.assertEqual(utils.snake_case("userId and-more"), "user_id_and_more")

    def test_safe_div(self):
        self.assertEqual(utils.safe_div(1, 0), 0.0)
        self.assertEqual(utils.safe_div(1, 4), 0.25)

    def test_group_by_and_first(self):
        self.assertEqual(utils.group_by([1, 2, 3, 4], lambda n: n % 2), {1: [1, 3], 0: [2, 4]})
        self.assertEqual(utils.first([1, 2, 3], lambda n: n > 1), 2)
        self.assertEqual(utils.first([], default="none"), "none")

    def test_merge_dicts(self):
        self.assertEqual(utils.merge_dicts({"a": 1}, {"a": 2, "b": 3}, c=4), {"a": 2, "b": 3, "c": 4})

    def test_retry(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise OSError("try again")
            return "ok"

        self.assertEqual(utils.retry(flaky), "ok")
        with self.assertRaises(KeyError):
            utils.retry(lambda: {}["x"], attempts=2)
'''

TD_TASK = """Add type hints and a one-line docstring to every public function in utils.py (the functions whose names do not start with "_"). Every parameter, *args and **kwargs included, and every return value needs an annotation. Use the typing module (for example Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar) where it helps; the module must still import on Python 3.9. Each docstring is exactly one line.

Do not change any function's behaviour, body, name, parameters or defaults, and do not add or remove functions. Done means the check command passes."""

TD_HIDDEN = r'''import ast
import copy
import unittest

ORIGINAL = @ORIGINAL@


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def all_args(fn):
    a = fn.args
    return list(getattr(a, "posonlyargs", [])) + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]


def without_hints_and_docstring(fn):
    fn = copy.deepcopy(fn)
    fn.returns = None
    for arg in all_args(fn):
        arg.annotation = None
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        fn.body = body[1:]
    return ast.dump(fn)


class TypesDocsHidden(unittest.TestCase):
    def setUp(self):
        with open("utils.py", encoding="utf-8") as f:
            self.current = functions(f.read())
        self.original = functions(ORIGINAL)

    def test_same_functions(self):
        self.assertEqual(sorted(self.current), sorted(self.original))

    def test_public_functions_fully_annotated(self):
        for name, fn in self.current.items():
            if name.startswith("_"):
                continue
            with self.subTest(function=name):
                self.assertIsNotNone(fn.returns, "missing return annotation")
                missing = [a.arg for a in all_args(fn) if a.annotation is None]
                self.assertEqual(missing, [], "parameters without annotations")

    def test_public_functions_have_one_line_docstrings(self):
        for name, fn in self.current.items():
            if name.startswith("_"):
                continue
            with self.subTest(function=name):
                doc = ast.get_docstring(fn)
                self.assertTrue(doc, "missing docstring")
                self.assertNotIn("\n", doc.strip(), "docstring is not one line")

    def test_bodies_and_defaults_unchanged(self):
        for name, fn in self.original.items():
            with self.subTest(function=name):
                self.assertIn(name, self.current)
                self.assertEqual(without_hints_and_docstring(self.current[name]),
                                 without_hints_and_docstring(fn))

    def test_no_pep604_unions_without_future_import(self):
        # "X | Y" in an annotation is evaluated at import time and breaks Python 3.9, unless
        # "from __future__ import annotations" turns annotations into strings.
        with open("utils.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        if any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names) for n in tree.body):
            return
        for name, fn in self.current.items():
            annotations = [fn.returns] + [a.annotation for a in all_args(fn)]
            unions = [n for ann in annotations if ann is not None for n in ast.walk(ann)
                      if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)]
            self.assertEqual(unions, [], "%s uses X | Y in an annotation (breaks Python 3.9)" % name)

    def test_module_imports(self):
        import utils
        self.assertEqual(utils.chunk([1, 2, 3], 2), [[1, 2], [3]])


if __name__ == "__main__":
    unittest.main()
'''.replace("@ORIGINAL@", repr(UTILS))

TD_REF_UTILS = r'''"""Small helpers shared across the service."""
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Type, TypeVar

T = TypeVar("T")
K = TypeVar("K")

_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


def chunk(items: Sequence[T], size: int) -> List[Sequence[T]]:
    """Split items into consecutive slices of at most size elements."""
    if size < 1:
        raise ValueError("size must be at least 1")
    return [items[i:i + size] for i in range(0, len(items), size)]


def flatten(nested: Sequence[Any]) -> List[Any]:
    """Flatten nested lists and tuples into one flat list."""
    out = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


def dedupe(items: Sequence[Hashable]) -> List[Hashable]:
    """Return items without duplicates, keeping first-seen order."""
    return list(OrderedDict.fromkeys(items))


def clamp(value: float, low: float, high: float) -> float:
    """Limit value to the range low..high."""
    if low > high:
        raise ValueError("low must not be greater than high")
    return max(low, min(high, value))


def parse_bool(text: str) -> bool:
    """Parse a yes/no style string into a bool."""
    lowered = text.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off", ""):
        return False
    raise ValueError("not a boolean: %r" % text)


def _words(name):
    return re.split(r"[\s\-_]+", _CAMEL_BOUNDARY.sub(r"\1 \2", name).strip())


def snake_case(name: str) -> str:
    """Convert a camelCase, spaced or hyphenated name to snake_case."""
    return "_".join(word.lower() for word in _words(name) if word)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Divide a by b, returning default when b is zero."""
    return a / b if b else default


def group_by(items: Sequence[T], key: Callable[[T], K]) -> Dict[K, List[T]]:
    """Group items into lists keyed by key(item)."""
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups


def first(items: Sequence[T], predicate: Optional[Callable[[T], bool]] = None,
          default: Optional[T] = None) -> Optional[T]:
    """Return the first item matching predicate, or default."""
    for item in items:
        if predicate is None or predicate(item):
            return item
    return default


def merge_dicts(*dicts: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Merge dicts left to right, then apply keyword overrides."""
    merged = {}
    for d in dicts:
        merged.update(d)
    merged.update(overrides)
    return merged


def retry(func: Callable[[], T], attempts: int = 3,
          exceptions: Tuple[Type[BaseException], ...] = (Exception,)) -> T:
    """Call func up to attempts times, re-raising the last listed exception."""
    last = None
    for _ in range(attempts):
        try:
            return func()
        except exceptions as exc:
            last = exc
    raise last
'''

ADD_TYPES_DOCS = Case(
    name="add_types_docs",
    category="add_types_docs",
    task=TD_TASK,
    check=UNITTEST_TESTS_DIR,
    files={"utils.py": UTILS, "tests/__init__.py": "", "tests/test_utils.py": UTILS_TESTS},
    hint_files=["utils.py"],
    hidden=Hidden({"eval_hidden_add_types_docs.py": TD_HIDDEN},
                  "python3 -m unittest -q eval_hidden_add_types_docs tests.test_utils"),
    reference={"utils.py": TD_REF_UTILS},
)


# =============================================================================================
# 8. debug_symptom
# =============================================================================================

DS_SOURCES = r'''"""Where sales rows come from. In production this reads the warehouse; tests pass a dict."""


class StaticSource:
    """A source backed by a dict of day -> list of rows ({"region": str, "amount": int cents})."""

    def __init__(self, rows_by_day):
        self._rows_by_day = rows_by_day
        self.fetches = 0

    def sales(self, day):
        self.fetches += 1
        return [dict(row) for row in self._rows_by_day.get(day, [])]
'''

DS_REPORT = r'''"""Revenue numbers per region for the dashboard."""
from collections import defaultdict


def _accumulate(totals, rows):
    for row in rows:
        totals[row["region"]] += row["amount"]
    return totals


class RevenueReport:
    """Per-day revenue figures. Source rows are fetched once per day and cached."""

    def __init__(self, source):
        self._source = source
        self._rows_cache = {}
        self._by_day = {}

    def rows(self, day):
        if day not in self._rows_cache:
            self._rows_cache[day] = self._source.sales(day)
        return self._rows_cache[day]

    def totals(self, day):
        """Return {region: cents} for day."""
        totals = self._by_day.setdefault(day, defaultdict(int))
        return dict(_accumulate(totals, self.rows(day)))

    def grand_total(self, day):
        return sum(self.totals(day).values())

    def summary(self, day):
        totals = self.totals(day)
        regions = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
        return {"day": day, "regions": regions, "grand_total": sum(totals.values())}
'''

DS_FORMATTING = r'''"""Display helpers for money amounts."""


def dollars(cents):
    sign = "-" if cents < 0 else ""
    whole, rest = divmod(abs(cents), 100)
    return "%s$%s.%02d" % (sign, "{:,}".format(whole), rest)
'''

DS_DASHBOARD = r'''"""Text rendering of the revenue dashboard."""
from metrics.formatting import dollars


def render(report, day):
    summary = report.summary(day)
    lines = ["Revenue for %s" % day]
    for region, cents in summary["regions"]:
        lines.append("  %-8s %12s" % (region, dollars(cents)))
    lines.append("  %-8s %12s" % ("total", dollars(summary["grand_total"])))
    return "\n".join(lines)
'''

DS_TESTS = r'''import unittest

from dashboard import render
from metrics.report import RevenueReport
from metrics.sources import StaticSource

DAY = "2024-05-01"
ROWS = {
    DAY: [{"region": "north", "amount": 1250}, {"region": "south", "amount": 800},
          {"region": "north", "amount": 250}],
    "2024-05-02": [{"region": "east", "amount": 4000}],
}


class RevenueReportTest(unittest.TestCase):
    def make(self):
        return RevenueReport(StaticSource(ROWS))

    def test_totals(self):
        self.assertEqual(self.make().totals(DAY), {"north": 1500, "south": 800})

    def test_grand_total(self):
        self.assertEqual(self.make().grand_total(DAY), 2300)

    def test_summary_sorted_by_revenue(self):
        summary = self.make().summary(DAY)
        self.assertEqual(summary["regions"], [("north", 1500), ("south", 800)])
        self.assertEqual(summary["grand_total"], 2300)

    def test_rows_fetched_once(self):
        source = StaticSource(ROWS)
        report = RevenueReport(source)
        report.rows(DAY)
        report.rows(DAY)
        self.assertEqual(source.fetches, 1)

    def test_empty_day(self):
        self.assertEqual(self.make().totals("2024-01-01"), {})

    def test_render(self):
        text = render(self.make(), DAY)
        self.assertIn("north", text)
        self.assertIn("$23.00", text)
'''

DS_TASK = """Bug report from the dashboard team:

"The revenue dashboard shows the right per-region numbers when a day is first opened. After the page auto-refreshes (it calls dashboard.render(report, day) again with the same RevenueReport object), every region and the grand total show exactly twice the real revenue, and after another refresh three times. Restarting the server makes it right again until the next refresh."

Find the root cause and fix it. Repeated calls for the same day must keep returning the correct numbers, and source rows must still be fetched only once per day. Keep the public API (RevenueReport and its methods, dashboard.render) the same. Add a regression test to tests/test_report.py that fails without your fix. Done means the check command passes (it passes today too: the existing tests miss the bug)."""

DS_HIDDEN = r'''import unittest

from dashboard import render
from metrics.report import RevenueReport
from metrics.sources import StaticSource

DAY = "2024-05-01"
OTHER = "2024-05-02"
ROWS = {
    DAY: [{"region": "north", "amount": 1250}, {"region": "south", "amount": 800},
          {"region": "north", "amount": 250}],
    OTHER: [{"region": "east", "amount": 4000}],
}
EXPECTED = {"north": 1500, "south": 800}


class SymptomHidden(unittest.TestCase):
    def setUp(self):
        self.source = StaticSource(ROWS)
        self.report = RevenueReport(self.source)

    def test_render_is_stable(self):
        first = render(self.report, DAY)
        self.assertIn("$23.00", first)
        self.assertEqual(render(self.report, DAY), first)
        self.assertEqual(render(self.report, DAY), first)

    def test_repeated_summary(self):
        for _ in range(3):
            summary = self.report.summary(DAY)
            self.assertEqual(summary["regions"], [("north", 1500), ("south", 800)])
            self.assertEqual(summary["grand_total"], 2300)

    def test_mixed_calls(self):
        self.report.summary(DAY)
        self.assertEqual(self.report.totals(DAY), EXPECTED)
        self.assertEqual(self.report.grand_total(DAY), 2300)
        self.assertEqual(self.report.totals(DAY), EXPECTED)

    def test_days_interleaved(self):
        self.report.totals(DAY)
        self.report.totals(OTHER)
        self.assertEqual(self.report.totals(DAY), EXPECTED)
        self.assertEqual(self.report.totals(OTHER), {"east": 4000})

    def test_rows_still_fetched_once_per_day(self):
        for _ in range(3):
            self.report.summary(DAY)
            self.report.totals(OTHER)
        self.assertEqual(self.source.fetches, 2)

    def test_caller_mutation_does_not_leak(self):
        totals = self.report.totals(DAY)
        totals["north"] = 0
        self.assertEqual(self.report.totals(DAY), EXPECTED)


if __name__ == "__main__":
    unittest.main()
'''

DS_REF_TEST = '''

class RepeatedCallsTest(unittest.TestCase):
    def test_repeated_render_is_stable(self):
        report = RevenueReport(StaticSource(ROWS))
        first = render(report, DAY)
        self.assertEqual(render(report, DAY), first)
'''

DEBUG_SYMPTOM = Case(
    name="debug_symptom",
    category="debug_symptom",
    task=DS_TASK,
    check=UNITTEST_TESTS_DIR,
    files={
        "metrics/__init__.py": "",
        "metrics/sources.py": DS_SOURCES,
        "metrics/report.py": DS_REPORT,
        "metrics/formatting.py": DS_FORMATTING,
        "dashboard.py": DS_DASHBOARD,
        "tests/__init__.py": "",
        "tests/test_report.py": DS_TESTS,
    },
    hint_files=[],
    hidden=Hidden({"eval_hidden_debug_symptom.py": _with_model_tests_guard(
        DS_HIDDEN, "tests/test_report.py", DS_TESTS, {"metrics/report.py": DS_REPORT})},
                  "python3 -m unittest -q eval_hidden_debug_symptom tests.test_report"),
    reference={
        "metrics/report.py": _replace_once(
            "        totals = self._by_day.setdefault(day, defaultdict(int))\n"
            "        return dict(_accumulate(totals, self.rows(day)))\n",
            "        return dict(_accumulate(defaultdict(int), self.rows(day)))\n"),
        "tests/test_report.py": lambda text: text + DS_REF_TEST,
    },
)


CASES = [SPEC_MODULE, FOLLOW_PATTERN, RENAME_ACROSS_FILES, WRITE_TESTS, FIX_FAILING_TEST,
         LARGE_FILE_EDIT, ADD_TYPES_DOCS, DEBUG_SYMPTOM]
CASES_BY_NAME = {case.name: case for case in CASES}
