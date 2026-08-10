//! `#[derive(FdlArgs)]` -- proc-macro derive for flodl-cli's argv parser.
//!
//! This crate is re-exported by [`flodl-cli`](https://crates.io/crates/flodl-cli)
//! as `flodl_cli::FdlArgs`, so downstream binaries depend on `flodl-cli`,
//! not on this crate directly.
//!
//! The derive turns a plain struct with named fields into an argv parser
//! plus JSON schema emitter plus ANSI-coloured help renderer. One struct
//! is the single source of truth: doc-comments become help text,
//! attribute metadata becomes schema, field types become typed values.
//!
//! # Field attributes
//!
//! Each field carries exactly one of `#[option(...)]` (named flag,
//! kebab-cased from the field ident) or `#[arg(...)]` (positional).
//! The field type determines cardinality:
//!
//! - `bool` -- absent = `false`, present = `true`. `#[option]` only.
//! - `T` -- scalar, required. `#[option]` must supply `default = "..."`.
//! - `Option<T>` -- scalar, optional. Absent = `None`.
//! - `Vec<T>` -- `#[option]`: repeatable. `#[arg]`: variadic, last.
//!
//! Supported keys for `#[option]`: `short`, `default`, `choices`, `env`,
//! `completer`. For `#[arg]`: `default`, `choices`, `variadic`,
//! `completer`. Reserved flags (`--help`, `--version`, `--quiet`,
//! `--env`, and their shorts) cannot be shadowed; collisions error at
//! derive time.
//!
//! # Example
//!
//! The example below depends on the `flodl-cli` crate; it is marked
//! `ignore` because this crate is a proc-macro and doesn't depend on
//! `flodl-cli` itself. Copy the snippet into a `flodl-cli`-depending
//! binary to try it.
//!
//! ```ignore
//! use flodl_cli::{FdlArgs, parse_or_schema};
//!
//! /// Train a model.
//! #[derive(FdlArgs, Debug)]
//! struct TrainArgs {
//!     /// Model architecture to use.
//!     #[option(short = 'm', choices = &["mlp", "resnet"], default = "mlp")]
//!     model: String,
//!
//!     /// Number of epochs.
//!     #[option(short = 'e', default = "10")]
//!     epochs: u32,
//!
//!     /// API key, read from env if flag is absent.
//!     #[option(env = "WANDB_API_KEY")]
//!     wandb_key: Option<String>,
//!
//!     /// Extra dataset paths.
//!     #[arg(variadic)]
//!     datasets: Vec<String>,
//! }
//!
//! fn main() {
//!     let args: TrainArgs = parse_or_schema();
//!     // `--help` and `--fdl-schema` are intercepted by parse_or_schema.
//!     let _ = args;
//! }
//! ```
//!
//! # Enum form (subcommands)
//!
//! Deriving on an **enum of newtype variants** turns each variant into a
//! subcommand. The derive is a thin dispatcher: it peels the leading
//! subcommand token and delegates parsing, schema, and help to the
//! wrapped type, which carries its own `#[derive(FdlArgs)]`. No field
//! parsing happens at the enum level.
//!
//! ```ignore
//! use flodl_cli::{FdlArgs, parse_or_schema};
//!
//! /// flodl letter CLI.
//! #[derive(FdlArgs)]
//! enum Cli {
//!     /// Train a letter model on a dataset
//!     Train(TrainArgs),
//!     /// Evaluate a trained letter model on a test split
//!     Eval(EvalArgs),
//!     /// Generate samples (subcommand renamed from the variant ident)
//!     #[command(name = "gen")]
//!     Generate(GenArgs),
//! }
//!
//! fn main() {
//!     match parse_or_schema::<Cli>() {
//!         Cli::Train(a) => { /* ... */ let _ = a; }
//!         Cli::Eval(a) => { let _ = a; }
//!         Cli::Generate(a) => { let _ = a; }
//!     }
//! }
//! ```
//!
//! - Subcommand name = the variant ident kebab-cased (`TrainSubscan` →
//!   `train-subscan`), overridable with `#[command(name = "...")]`.
//! - Variant doc-comments become the per-subcommand descriptions shown in
//!   the parent `--help` command list.
//! - `--help` is contextual: `<bin> train --help` shows train's flags,
//!   `<bin> --help` shows the command list. `--fdl-schema` emits the full
//!   tree.
//! - Only single-tuple (newtype) variants are supported — each subcommand
//!   *is* a struct. Unit, named-field, and multi-field variants are
//!   rejected at derive time. A variant may itself wrap another
//!   `FdlArgs` enum for nested subcommands.
//!
//! See the [`flodl-cli`](https://docs.rs/flodl-cli) crate for the
//! user-facing API (`parse_or_schema`, `FdlArgsTrait`, `Schema`) and
//! the full CLI reference.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, quote_spanned};
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Fields, GenericArgument, Ident, Lit,
    PathArguments, Type, TypePath, parse_macro_input,
};

// ── Reserved flags (kept in sync with flodl-cli/src/config.rs) ─────────

const RESERVED_LONGS: &[&str] = &["help", "version", "quiet", "env"];
const RESERVED_SHORTS: &[char] = &['h', 'V', 'q', 'v', 'e'];

// ── Entry point ─────────────────────────────────────────────────────────

/// Derive `FdlArgs` to generate an argv parser, `--fdl-schema` JSON
/// emitter, and ANSI-coloured `--help` renderer.
///
/// On a **struct with named fields**, each field is a flag or positional.
/// On an **enum of newtype variants**, each variant is a subcommand that
/// delegates to the wrapped type. See the [crate-level docs](crate) for
/// the attribute reference and worked examples of both forms.
#[proc_macro_derive(FdlArgs, attributes(option, arg, command))]
pub fn derive_fdl_args(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match impl_derive(input) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error().into(),
    }
}

fn impl_derive(input: DeriveInput) -> syn::Result<TokenStream> {
    let ident = &input.ident;
    let description = extract_doc(&input.attrs);

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => &n.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    ident,
                    "FdlArgs requires a struct with named fields",
                ));
            }
        },
        // An enum is a variant-shaped CLI: each newtype variant is a
        // subcommand wrapping a type that itself derives FdlArgs. The
        // derive here is a thin dispatcher over those inner impls.
        Data::Enum(e) => return impl_enum_derive(ident, description.as_deref(), e),
        _ => {
            return Err(syn::Error::new_spanned(
                ident,
                "FdlArgs requires a struct or enum",
            ));
        }
    };

    let mut parsed: Vec<FieldSpec> = Vec::new();
    for f in fields {
        parsed.push(parse_field(f)?);
    }

    validate_collisions(&parsed)?;

    let spec_build = build_spec_expr(&parsed);
    let schema_build = build_schema_expr(&parsed, description.as_deref());
    let extract = build_extractor(ident, &parsed)?;
    let render_help = build_help_expr(&parsed, description.as_deref(), &ident.to_string());
    let env_injection = build_env_injection(&parsed);

    let expanded = quote! {
        impl ::flodl_cli::FdlArgsTrait for #ident {
            fn try_parse_from(args: &[::std::string::String])
                -> ::std::result::Result<Self, ::std::string::String>
            {
                let spec = #spec_build;
                #env_injection
                let parsed = ::flodl_cli::args::parser::parse(&spec, args)?;
                #extract
            }

            fn schema() -> ::flodl_cli::Schema {
                #schema_build
            }

            fn render_help() -> ::std::string::String {
                #render_help
            }
        }
    };
    Ok(expanded.into())
}

// ── Enum (variant-shaped CLI) ───────────────────────────────────────────

/// What we learn from one enum variant: a subcommand.
struct VariantSpec {
    /// Variant identifier (used in the generated `Self::Ident(...)` arm).
    ident: Ident,
    /// Subcommand name on the command line — kebab of the ident, or the
    /// `#[command(name = "...")]` override.
    name: String,
    /// The wrapped type (`TrainArgs` in `Train(TrainArgs)`); must itself
    /// implement `FdlArgsTrait` (i.e. derive `FdlArgs`).
    inner_ty: Type,
    /// Variant doc-comment → the subcommand's one-line help description.
    description: Option<String>,
}

/// Generate the `FdlArgsTrait` impl for an enum of newtype variants. The
/// impl is a dispatcher: it peels the leading subcommand token and
/// delegates parse / schema / help to the wrapped type, which carries its
/// own derived impl. No field parsing happens here.
fn impl_enum_derive(
    ident: &Ident,
    description: Option<&str>,
    data: &syn::DataEnum,
) -> syn::Result<TokenStream> {
    if data.variants.is_empty() {
        return Err(syn::Error::new_spanned(
            ident,
            "FdlArgs enum needs at least one variant (each variant is a subcommand)",
        ));
    }

    let mut variants: Vec<VariantSpec> = Vec::new();
    let mut seen: std::collections::HashMap<String, Span> = std::collections::HashMap::new();
    for v in &data.variants {
        let inner_ty = match &v.fields {
            Fields::Unnamed(f) if f.unnamed.len() == 1 => f.unnamed[0].ty.clone(),
            _ => {
                return Err(syn::Error::new_spanned(
                    v,
                    "FdlArgs enum variants must be single-tuple (newtype) variants \
                     wrapping a type that derives FdlArgs, e.g. `Train(TrainArgs)` \
                     (unit, named-field, and multi-field variants are not supported)",
                ));
            }
        };
        let name = variant_command_name(v)?;
        if let Some(prev) = seen.insert(name.clone(), v.span()) {
            let _ = prev;
            return Err(syn::Error::new_spanned(
                v,
                format!("duplicate subcommand name `{name}`"),
            ));
        }
        variants.push(VariantSpec {
            ident: v.ident.clone(),
            name,
            inner_ty,
            description: extract_doc(&v.attrs),
        });
    }

    // Comma-separated name list for "expected one of" / "did you mean".
    let names: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
    let names_csv = names.join(", ");
    let names_arr = quote! { &[ #( #names ),* ] };

    // try_parse_from: match the subcommand token, delegate the tail.
    let parse_arms = variants.iter().map(|v| {
        let name = &v.name;
        let vident = &v.ident;
        let inner_ty = &v.inner_ty;
        quote! {
            #name => ::std::result::Result::Ok(#ident::#vident(
                <#inner_ty as ::flodl_cli::FdlArgsTrait>::try_parse_from(&args[1..])?
            )),
        }
    });

    // schema(): build a branch node, one child per subcommand.
    let schema_inserts = variants.iter().map(|v| {
        let name = &v.name;
        let inner_ty = &v.inner_ty;
        let desc_set = match &v.description {
            Some(d) => quote! { __child.description = ::std::option::Option::Some(::std::string::String::from(#d)); },
            None => quote! {},
        };
        quote! {
            {
                let mut __child = <#inner_ty as ::flodl_cli::FdlArgsTrait>::schema();
                #desc_set
                __commands.insert(::std::string::String::from(#name), __child);
            }
        }
    });

    // render_help_path(): peel the first non-flag token, delegate to that
    // subcommand's (recursive) help; otherwise the command list.
    let help_path_arms = variants.iter().map(|v| {
        let name = &v.name;
        let inner_ty = &v.inner_ty;
        quote! {
            #name => return <#inner_ty as ::flodl_cli::FdlArgsTrait>::render_help_path(__tail),
        }
    });

    // render_help(): the root command listing.
    let banner = description
        .map(str::to_string)
        .unwrap_or_else(|| ident.to_string());
    let command_lines = variants.iter().map(|v| {
        let label = v.name.clone();
        let pad = " ".repeat(36usize.saturating_sub(4 + label.chars().count()));
        let tail = v.description.clone().unwrap_or_default();
        quote! {
            out.push_str("    ");
            out.push_str(&::flodl_cli::style::green(#label));
            out.push_str(#pad);
            out.push_str(#tail);
            out.push('\n');
        }
    });

    let expanded = quote! {
        impl ::flodl_cli::FdlArgsTrait for #ident {
            fn try_parse_from(args: &[::std::string::String])
                -> ::std::result::Result<Self, ::std::string::String>
            {
                let sub = match args.get(1) {
                    ::std::option::Option::Some(s) => s.as_str(),
                    ::std::option::Option::None => {
                        return ::std::result::Result::Err(::std::format!(
                            "missing command, expected one of: {}", #names_csv
                        ));
                    }
                };
                match sub {
                    #( #parse_arms )*
                    other => {
                        match ::flodl_cli::args::parser::suggest(#names_arr, other) {
                            ::std::option::Option::Some(s) => ::std::result::Result::Err(::std::format!(
                                "unknown command `{other}`, did you mean `{s}`?"
                            )),
                            ::std::option::Option::None => ::std::result::Result::Err(::std::format!(
                                "unknown command `{other}`, expected one of: {}", #names_csv
                            )),
                        }
                    }
                }
            }

            fn schema() -> ::flodl_cli::Schema {
                let mut __commands: ::std::collections::BTreeMap<::std::string::String, ::flodl_cli::Schema> =
                    ::std::collections::BTreeMap::new();
                #( #schema_inserts )*
                ::flodl_cli::Schema {
                    commands: __commands,
                    ..::core::default::Default::default()
                }
            }

            fn render_help() -> ::std::string::String {
                let mut out = ::flodl_cli::help::banner_string(#banner);
                out.push_str(&::flodl_cli::style::yellow("Commands"));
                out.push_str(":\n");
                #( #command_lines )*
                out
            }

            fn render_help_path(args: &[::std::string::String]) -> ::std::string::String {
                // The subcommand is args[1] (mirrors `try_parse_from`). A
                // leading flag or no token falls to the top-level command list.
                // Scanning past leading flags to the first bare token would
                // mis-pick an option value (e.g. `42` in `--seed 42 train`) as
                // the subcommand.
                if let ::std::option::Option::Some(__sub) =
                    args.get(1).filter(|s| !s.starts_with('-'))
                {
                    let __tail = &args[1..];
                    match __sub.as_str() {
                        #( #help_path_arms )*
                        _ => {}
                    }
                }
                Self::render_help()
            }
        }
    };
    Ok(expanded.into())
}

/// Resolve a variant's subcommand name: kebab of the ident by default, or
/// the `#[command(name = "...")]` override.
fn variant_command_name(v: &syn::Variant) -> syn::Result<String> {
    for attr in &v.attrs {
        if !attr.path().is_ident("command") {
            continue;
        }
        let mut name: Option<String> = None;
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let s: syn::LitStr = meta.value()?.parse()?;
                name = Some(s.value());
                Ok(())
            } else {
                Err(meta.error("unknown #[command] key (valid: name)"))
            }
        })?;
        if let Some(n) = name {
            if n.trim().is_empty() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "#[command(name = ...)] must be non-empty",
                ));
            }
            return Ok(n);
        }
    }
    Ok(pascal_to_kebab(&v.ident.to_string()))
}

// ── Field spec (what we learn from each field) ──────────────────────────

#[derive(Clone)]
enum FieldKind {
    Option,
    Arg,
}

#[derive(Clone)]
enum TypeShape {
    /// `bool`
    Bool,
    /// `T` — scalar
    Scalar,
    /// `Option<T>`
    Opt,
    /// `Vec<T>`
    List,
}

#[derive(Clone)]
struct FieldSpec {
    ident: Ident,
    kind: FieldKind,
    shape: TypeShape,
    /// The "inner" type (for `Option<T>` / `Vec<T>`, the `T`; for `T`, `T` itself).
    inner_ty: Type,
    description: Option<String>,
    // Attribute contents
    short: Option<char>,
    default: Option<String>,
    choices: Option<Vec<String>>,
    env: Option<String>,
    completer: Option<String>,
    variadic: bool,
    span: Span,
}

fn parse_field(f: &syn::Field) -> syn::Result<FieldSpec> {
    let ident = f
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new_spanned(f, "FdlArgs requires named fields"))?;
    let description = extract_doc(&f.attrs);
    let (shape, inner_ty) = classify_type(&f.ty);

    // Exactly one of #[option] / #[arg] must be present (plain fields
    // are NOT auto-treated as options in this MVP — explicit is better
    // while the contract settles).
    let mut kind: Option<FieldKind> = None;
    let mut short: Option<char> = None;
    let mut default: Option<String> = None;
    let mut choices: Option<Vec<String>> = None;
    let mut env: Option<String> = None;
    let mut completer: Option<String> = None;
    let mut variadic = false;

    for attr in &f.attrs {
        if attr.path().is_ident("option") {
            if kind.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "field cannot have both #[option] and #[arg]",
                ));
            }
            kind = Some(FieldKind::Option);
            parse_option_attr(
                attr,
                &mut short,
                &mut default,
                &mut choices,
                &mut env,
                &mut completer,
            )?;
        } else if attr.path().is_ident("arg") {
            if kind.is_some() {
                return Err(syn::Error::new_spanned(
                    attr,
                    "field cannot have both #[option] and #[arg]",
                ));
            }
            kind = Some(FieldKind::Arg);
            parse_arg_attr(
                attr,
                &mut default,
                &mut choices,
                &mut variadic,
                &mut completer,
            )?;
        }
    }

    let kind = kind.ok_or_else(|| {
        syn::Error::new_spanned(&ident, "field must carry either #[option] or #[arg]")
    })?;

    // Type + kind + attrs consistency checks.
    match kind {
        FieldKind::Option => {
            if matches!(shape, TypeShape::Bool) && default.is_some() {
                return Err(syn::Error::new_spanned(
                    &f.ty,
                    "#[option(default = ...)] is meaningless on a bool flag (absent=false, present=true)",
                ));
            }
            if matches!(shape, TypeShape::Bool) && env.is_some() {
                return Err(syn::Error::new_spanned(
                    &f.ty,
                    "#[option(env = ...)] is not supported on bare `bool` (truthy/falsy string semantics are ambiguous) — use `Option<bool>` if you need env fallback",
                ));
            }
            if matches!(shape, TypeShape::Scalar)
                && default.is_none()
                && !matches!(shape, TypeShape::Bool)
            {
                return Err(syn::Error::new_spanned(
                    &f.ty,
                    "#[option] on a non-Option, non-bool type requires `default = \"...\"` (the field must always have a value)",
                ));
            }
            if variadic {
                return Err(syn::Error::new_spanned(
                    &ident,
                    "`variadic` only applies to #[arg], not #[option]",
                ));
            }
        }
        FieldKind::Arg => {
            if matches!(shape, TypeShape::Bool) {
                return Err(syn::Error::new_spanned(
                    &f.ty,
                    "positional #[arg] cannot be a bool (positionals always carry a value)",
                ));
            }
            if short.is_some() {
                return Err(syn::Error::new_spanned(
                    &ident,
                    "`short` cannot be used on #[arg] (positionals have no short form)",
                ));
            }
            if variadic && !matches!(shape, TypeShape::List) {
                return Err(syn::Error::new_spanned(
                    &f.ty,
                    "#[arg(variadic)] requires a Vec<T> field",
                ));
            }
        }
    }

    Ok(FieldSpec {
        ident,
        kind,
        shape,
        inner_ty,
        description,
        short,
        default,
        choices,
        env,
        completer,
        variadic,
        span: f.span(),
    })
}

// ── Attribute parsing ───────────────────────────────────────────────────

fn parse_option_attr(
    attr: &Attribute,
    short: &mut Option<char>,
    default: &mut Option<String>,
    choices: &mut Option<Vec<String>>,
    env: &mut Option<String>,
    completer: &mut Option<String>,
) -> syn::Result<()> {
    if matches!(attr.meta, syn::Meta::Path(_)) {
        return Ok(()); // bare #[option]
    }
    attr.parse_nested_meta(|meta| {
        let key = meta
            .path
            .get_ident()
            .ok_or_else(|| meta.error("expected identifier key in #[option]"))?;
        match key.to_string().as_str() {
            "short" => {
                let v: syn::LitChar = meta.value()?.parse()?;
                *short = Some(v.value());
            }
            "default" => {
                let v: syn::LitStr = meta.value()?.parse()?;
                *default = Some(v.value());
            }
            "choices" => {
                *choices = Some(parse_choices(&meta)?);
            }
            "env" => {
                let v: syn::LitStr = meta.value()?.parse()?;
                *env = Some(v.value());
            }
            "completer" => {
                let v: syn::LitStr = meta.value()?.parse()?;
                *completer = Some(v.value());
            }
            other => {
                return Err(meta.error(format!(
                    "unknown #[option] attribute `{other}` (valid: short, default, choices, env, completer)"
                )));
            }
        }
        Ok(())
    })
}

fn parse_arg_attr(
    attr: &Attribute,
    default: &mut Option<String>,
    choices: &mut Option<Vec<String>>,
    variadic: &mut bool,
    completer: &mut Option<String>,
) -> syn::Result<()> {
    if matches!(attr.meta, syn::Meta::Path(_)) {
        return Ok(());
    }
    attr.parse_nested_meta(|meta| {
        let key = meta
            .path
            .get_ident()
            .ok_or_else(|| meta.error("expected identifier key in #[arg]"))?;
        match key.to_string().as_str() {
            "default" => {
                let v: syn::LitStr = meta.value()?.parse()?;
                *default = Some(v.value());
            }
            "choices" => {
                *choices = Some(parse_choices(&meta)?);
            }
            "variadic" => {
                // Either `variadic` alone or `variadic = true`.
                *variadic = true;
                if meta.input.peek(syn::Token![=]) {
                    let v: syn::LitBool = meta.value()?.parse()?;
                    *variadic = v.value();
                }
            }
            "completer" => {
                let v: syn::LitStr = meta.value()?.parse()?;
                *completer = Some(v.value());
            }
            other => {
                return Err(meta.error(format!(
                    "unknown #[arg] attribute `{other}` (valid: default, choices, variadic, completer)"
                )));
            }
        }
        Ok(())
    })
}

fn parse_choices(meta: &syn::meta::ParseNestedMeta) -> syn::Result<Vec<String>> {
    // Accept both `choices = &["a", "b"]` and `choices = ["a", "b"]`.
    let expr: Expr = meta.value()?.parse()?;
    let arr = match expr {
        Expr::Reference(r) => *r.expr,
        e => e,
    };
    match arr {
        Expr::Array(arr) => {
            let mut out = Vec::with_capacity(arr.elems.len());
            for e in arr.elems {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) = e
                {
                    out.push(s.value());
                } else {
                    return Err(syn::Error::new_spanned(
                        e,
                        "choices must be string literals",
                    ));
                }
            }
            Ok(out)
        }
        other => Err(syn::Error::new_spanned(
            other,
            "choices must be an array literal, e.g. `&[\"a\", \"b\"]`",
        )),
    }
}

// ── Type classification ─────────────────────────────────────────────────

fn classify_type(ty: &Type) -> (TypeShape, Type) {
    if let Type::Path(TypePath { path, .. }) = ty
        && let Some(seg) = path.segments.last()
    {
        let name = seg.ident.to_string();
        if name == "bool" {
            return (TypeShape::Bool, ty.clone());
        }
        if name == "Option"
            && let Some(inner) = first_generic(&seg.arguments)
        {
            return (TypeShape::Opt, inner);
        }
        if name == "Vec"
            && let Some(inner) = first_generic(&seg.arguments)
        {
            return (TypeShape::List, inner);
        }
    }
    (TypeShape::Scalar, ty.clone())
}

fn first_generic(args: &PathArguments) -> Option<Type> {
    if let PathArguments::AngleBracketed(a) = args {
        for arg in &a.args {
            if let GenericArgument::Type(t) = arg {
                return Some(t.clone());
            }
        }
    }
    None
}

// ── Validation ──────────────────────────────────────────────────────────

fn validate_collisions(fields: &[FieldSpec]) -> syn::Result<()> {
    let mut seen_long: std::collections::HashMap<String, Span> = std::collections::HashMap::new();
    let mut seen_short: std::collections::HashMap<char, Span> = std::collections::HashMap::new();

    // Positionals: variadic-last, no-required-after-optional.
    let mut seen_optional = false;
    for f in fields {
        if !matches!(f.kind, FieldKind::Arg) {
            continue;
        }
        let is_optional = matches!(f.shape, TypeShape::Opt) || f.default.is_some() || f.variadic;
        if seen_optional && !is_optional {
            return Err(syn::Error::new(
                f.span,
                "required positional cannot follow an optional one",
            ));
        }
        if is_optional {
            seen_optional = true;
        }
    }
    // Variadic may only be the last arg.
    let mut saw_variadic = false;
    for f in fields {
        if !matches!(f.kind, FieldKind::Arg) {
            continue;
        }
        if saw_variadic {
            return Err(syn::Error::new(
                f.span,
                "variadic positional must be the last one",
            ));
        }
        if f.variadic {
            saw_variadic = true;
        }
    }

    for f in fields {
        if !matches!(f.kind, FieldKind::Option) {
            continue;
        }
        let long = kebab(&f.ident.to_string());
        if RESERVED_LONGS.contains(&long.as_str()) {
            return Err(syn::Error::new(
                f.span,
                format!("--{long} shadows a reserved fdl-level flag"),
            ));
        }
        if let Some(prev) = seen_long.insert(long.clone(), f.span) {
            return Err(syn::Error::new(
                f.span,
                format!(
                    "duplicate long flag --{long} (previously declared at {:?})",
                    prev
                ),
            ));
        }
        if let Some(s) = f.short {
            if RESERVED_SHORTS.contains(&s) {
                return Err(syn::Error::new(
                    f.span,
                    format!("-{s} shadows a reserved fdl-level flag"),
                ));
            }
            if let Some(prev) = seen_short.insert(s, f.span) {
                return Err(syn::Error::new(
                    f.span,
                    format!("duplicate short -{s} (previously declared at {:?})", prev),
                ));
            }
        }
    }

    Ok(())
}

// ── Code generators ─────────────────────────────────────────────────────

fn build_spec_expr(fields: &[FieldSpec]) -> TokenStream2 {
    let opts = fields
        .iter()
        .filter(|f| matches!(f.kind, FieldKind::Option))
        .map(build_option_decl);
    let positionals = fields
        .iter()
        .filter(|f| matches!(f.kind, FieldKind::Arg))
        .map(build_positional_decl);

    quote! {
        ::flodl_cli::args::parser::ArgsSpec {
            options: vec![ #( #opts ),* ],
            positionals: vec![ #( #positionals ),* ],
            // Derive-parsed CLIs are authoritative about their own
            // surface — unknown flags are programmer errors, not
            // legitimate pass-through. Stay strict.
            lenient_unknowns: false,
        }
    }
}

fn build_option_decl(f: &FieldSpec) -> TokenStream2 {
    let long = kebab(&f.ident.to_string());
    let takes_value = !matches!(f.shape, TypeShape::Bool);
    let allows_bare = match f.shape {
        TypeShape::Bool => true,
        _ => f.default.is_some(),
    };
    let repeatable = matches!(f.shape, TypeShape::List);
    let short_expr = match f.short {
        Some(c) => quote! { ::std::option::Option::Some(#c) },
        None => quote! { ::std::option::Option::None },
    };
    let choices_expr = match &f.choices {
        Some(list) => {
            let elems = list.iter();
            quote! { ::std::option::Option::Some(vec![ #( ::std::string::String::from(#elems) ),* ]) }
        }
        None => quote! { ::std::option::Option::None },
    };

    quote! {
        ::flodl_cli::args::parser::OptionDecl {
            long: ::std::string::String::from(#long),
            short: #short_expr,
            takes_value: #takes_value,
            allows_bare: #allows_bare,
            repeatable: #repeatable,
            choices: #choices_expr,
        }
    }
}

fn build_positional_decl(f: &FieldSpec) -> TokenStream2 {
    let name = kebab(&f.ident.to_string());
    let required = matches!(f.shape, TypeShape::Scalar) && f.default.is_none() && !f.variadic;
    let variadic = f.variadic;
    let choices_expr = match &f.choices {
        Some(list) => {
            let elems = list.iter();
            quote! { ::std::option::Option::Some(vec![ #( ::std::string::String::from(#elems) ),* ]) }
        }
        None => quote! { ::std::option::Option::None },
    };
    quote! {
        ::flodl_cli::args::parser::PositionalDecl {
            name: ::std::string::String::from(#name),
            required: #required,
            variadic: #variadic,
            choices: #choices_expr,
        }
    }
}

fn build_schema_expr(fields: &[FieldSpec], description: Option<&str>) -> TokenStream2 {
    let desc_expr = match description {
        Some(d) => quote! { ::std::option::Option::Some(::std::string::String::from(#d)) },
        None => quote! { ::std::option::Option::None },
    };

    let option_inserts = fields
        .iter()
        .filter(|f| matches!(f.kind, FieldKind::Option))
        .map(|f| {
            let long = kebab(&f.ident.to_string());
            let ty = schema_type_str(f);
            let desc_expr = match &f.description {
                Some(d) => quote! { ::std::option::Option::Some(::std::string::String::from(#d)) },
                None => quote! { ::std::option::Option::None },
            };
            let default_expr = match &f.default {
                Some(v) => quote! { ::std::option::Option::Some(::flodl_cli::serde_json::Value::String(::std::string::String::from(#v))) },
                None => quote! { ::std::option::Option::None },
            };
            let choices_expr = match &f.choices {
                Some(list) => {
                    let elems = list.iter();
                    quote! {
                        ::std::option::Option::Some(vec![
                            #( ::flodl_cli::serde_json::Value::String(::std::string::String::from(#elems)) ),*
                        ])
                    }
                }
                None => quote! { ::std::option::Option::None },
            };
            let short_expr = match f.short {
                Some(c) => {
                    let cs = c.to_string();
                    quote! { ::std::option::Option::Some(::std::string::String::from(#cs)) }
                }
                None => quote! { ::std::option::Option::None },
            };
            let env_expr = match &f.env {
                Some(v) => quote! { ::std::option::Option::Some(::std::string::String::from(#v)) },
                None => quote! { ::std::option::Option::None },
            };
            let completer_expr = match &f.completer {
                Some(v) => quote! { ::std::option::Option::Some(::std::string::String::from(#v)) },
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                options.insert(
                    ::std::string::String::from(#long),
                    ::flodl_cli::OptionSpec {
                        ty: ::std::string::String::from(#ty),
                        description: #desc_expr,
                        default: #default_expr,
                        choices: #choices_expr,
                        short: #short_expr,
                        env: #env_expr,
                        completer: #completer_expr,
                    },
                );
            }
        });

    let arg_pushes = fields
        .iter()
        .filter(|f| matches!(f.kind, FieldKind::Arg))
        .map(|f| {
            let name = kebab(&f.ident.to_string());
            let ty = schema_type_str(f);
            let desc_expr = match &f.description {
                Some(d) => quote! { ::std::option::Option::Some(::std::string::String::from(#d)) },
                None => quote! { ::std::option::Option::None },
            };
            let required = matches!(f.shape, TypeShape::Scalar) && f.default.is_none() && !f.variadic;
            let variadic = f.variadic;
            let default_expr = match &f.default {
                Some(v) => quote! { ::std::option::Option::Some(::flodl_cli::serde_json::Value::String(::std::string::String::from(#v))) },
                None => quote! { ::std::option::Option::None },
            };
            let choices_expr = match &f.choices {
                Some(list) => {
                    let elems = list.iter();
                    quote! {
                        ::std::option::Option::Some(vec![
                            #( ::flodl_cli::serde_json::Value::String(::std::string::String::from(#elems)) ),*
                        ])
                    }
                }
                None => quote! { ::std::option::Option::None },
            };
            let completer_expr = match &f.completer {
                Some(v) => quote! { ::std::option::Option::Some(::std::string::String::from(#v)) },
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                args.push(::flodl_cli::ArgSpec {
                    name: ::std::string::String::from(#name),
                    ty: ::std::string::String::from(#ty),
                    description: #desc_expr,
                    required: #required,
                    variadic: #variadic,
                    default: #default_expr,
                    choices: #choices_expr,
                    completer: #completer_expr,
                });
            }
        });

    // `desc_expr` is retained for future use (Schema may grow a
    // description field). Bind it to `_` only when it has a concrete
    // type — interpolating a bare `Option::None` into `let _ = ...;`
    // leaves rustc unable to infer the type parameter (E0282) when
    // the caller's surrounding context doesn't pin it down, which
    // happened inside test modules.
    let _ = desc_expr;
    quote! {
        {
            let mut options: ::std::collections::BTreeMap<::std::string::String, ::flodl_cli::OptionSpec> =
                ::std::collections::BTreeMap::new();
            let mut args: ::std::vec::Vec<::flodl_cli::ArgSpec> = ::std::vec::Vec::new();
            #( #option_inserts )*
            #( #arg_pushes )*
            ::flodl_cli::Schema {
                args,
                options,
                strict: false,
                // A `#[derive(FdlArgs)]` struct is always a leaf — no
                // subcommand tree. The enum derive builds branches itself.
                description: ::std::option::Option::None,
                commands: ::std::collections::BTreeMap::new(),
            }
        }
    }
}

fn schema_type_str(f: &FieldSpec) -> &'static str {
    let inner = inner_ty_name(&f.inner_ty);
    let base = match inner.as_str() {
        "bool" => "bool",
        "String" | "&str" => "string",
        "PathBuf" | "Path" => "path",
        "f32" | "f64" => "float",
        // Any integer-ish.
        "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" => "int",
        _ => "string",
    };
    match f.shape {
        TypeShape::List => match base {
            "string" => "list[string]",
            "int" => "list[int]",
            "float" => "list[float]",
            "path" => "list[path]",
            _ => "list[string]",
        },
        TypeShape::Bool => "bool",
        _ => base,
    }
}

fn inner_ty_name(ty: &Type) -> String {
    if let Type::Path(TypePath { path, .. }) = ty
        && let Some(seg) = path.segments.last()
    {
        return seg.ident.to_string();
    }
    String::from("_")
}

/// Emit an argv pre-processing block that, for each `#[option(env = "...")]`
/// field absent from argv, appends `--<long> <value>` sourced from the named
/// environment variable. After this runs, the standard parser pipeline
/// handles the value exactly like an argv-supplied flag — choices, strict
/// unknowns, and `FromStr` all fire unchanged.
///
/// Precedence (highest wins): argv flag → env var → `default`. Empty env
/// vars fall through (consistent with `FDL_ENV` handling in `main.rs`).
/// Boolean fields are rejected at derive time elsewhere, so we never
/// inject `--foo` without a value.
fn build_env_injection(fields: &[FieldSpec]) -> TokenStream2 {
    let mut injections: Vec<TokenStream2> = Vec::new();
    for f in fields {
        let Some(env_name) = f.env.as_deref() else {
            continue;
        };
        // Positional args (#[arg]) don't have an env path in this MVP —
        // they're typically required and rarely env-driven. Skip them.
        if matches!(f.kind, FieldKind::Arg) {
            continue;
        }
        let long = kebab(&f.ident.to_string());
        let long_flag = format!("--{long}");
        let long_eq_prefix = format!("--{long}=");
        let short_tok = match &f.short {
            Some(c) => {
                let short_exact = format!("-{c}");
                quote! {
                    || a.as_str() == #short_exact
                }
            }
            None => quote! {},
        };
        injections.push(quote! {
            {
                let has_flag = __env_args.iter().any(|a: &::std::string::String| {
                    a.as_str() == #long_flag
                        || a.as_str().starts_with(#long_eq_prefix)
                        #short_tok
                });
                if !has_flag {
                    if let ::std::result::Result::Ok(v) = ::std::env::var(#env_name) {
                        if !v.is_empty() {
                            __env_args.push(::std::string::String::from(#long_flag));
                            __env_args.push(v);
                        }
                    }
                }
            }
        });
    }
    if injections.is_empty() {
        return quote! {};
    }
    quote! {
        let __env_args: ::std::vec::Vec<::std::string::String> = {
            let mut __env_args: ::std::vec::Vec<::std::string::String> = args.to_vec();
            #( #injections )*
            __env_args
        };
        let args: &[::std::string::String] = &__env_args[..];
    }
}

fn build_extractor(ident: &Ident, fields: &[FieldSpec]) -> syn::Result<TokenStream2> {
    let mut field_inits: Vec<TokenStream2> = Vec::new();
    let mut positional_idx: usize = 0;
    for f in fields {
        match f.kind {
            FieldKind::Option => field_inits.push(option_extraction(f)),
            FieldKind::Arg => {
                field_inits.push(arg_extraction(f, positional_idx));
                if !f.variadic {
                    positional_idx += 1;
                }
            }
        }
    }
    let field_names: Vec<&Ident> = fields.iter().map(|f| &f.ident).collect();
    Ok(quote! {
        #( #field_inits )*
        ::std::result::Result::Ok(#ident {
            #( #field_names ),*
        })
    })
}

fn option_extraction(f: &FieldSpec) -> TokenStream2 {
    let ident = &f.ident;
    let long = kebab(&ident.to_string());
    let inner_ty = &f.inner_ty;
    let span = ident.span();
    let parse_one = quote_spanned! { span =>
        |s: &::std::string::String| -> ::std::result::Result<#inner_ty, ::std::string::String> {
            <#inner_ty as ::std::str::FromStr>::from_str(s)
                .map_err(|e| format!("--{}: {}", #long, e))
        }
    };

    match f.shape {
        TypeShape::Bool => quote! {
            let #ident: bool = matches!(
                parsed.options.get(#long),
                ::std::option::Option::Some(::flodl_cli::args::parser::OptionState::BarePresent)
            );
        },
        TypeShape::Scalar => {
            // Must have a default (validated earlier).
            let default_lit = f.default.as_deref().unwrap();
            quote! {
                let #ident: #inner_ty = match parsed.options.get(#long) {
                    ::std::option::Option::Some(::flodl_cli::args::parser::OptionState::WithValues(v)) => {
                        let s = &v[0];
                        (#parse_one)(s)?
                    }
                    _ => {
                        let s = ::std::string::String::from(#default_lit);
                        (#parse_one)(&s).expect("default value must parse")
                    }
                };
            }
        }
        TypeShape::Opt => {
            let default_tok = match &f.default {
                Some(v) => quote! { ::std::option::Option::Some({
                    let s = ::std::string::String::from(#v);
                    (#parse_one)(&s).expect("default value must parse")
                }) },
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                let #ident: ::std::option::Option<#inner_ty> = match parsed.options.get(#long) {
                    ::std::option::Option::Some(::flodl_cli::args::parser::OptionState::WithValues(v)) => {
                        ::std::option::Option::Some((#parse_one)(&v[0])?)
                    }
                    ::std::option::Option::Some(::flodl_cli::args::parser::OptionState::BarePresent) => {
                        #default_tok
                    }
                    ::std::option::Option::None => ::std::option::Option::None,
                };
            }
        }
        TypeShape::List => quote! {
            let #ident: ::std::vec::Vec<#inner_ty> = match parsed.options.get(#long) {
                ::std::option::Option::Some(::flodl_cli::args::parser::OptionState::WithValues(v)) => {
                    let mut out: ::std::vec::Vec<#inner_ty> = ::std::vec::Vec::with_capacity(v.len());
                    for s in v {
                        out.push((#parse_one)(s)?);
                    }
                    out
                }
                _ => ::std::vec::Vec::new(),
            };
        },
    }
}

fn arg_extraction(f: &FieldSpec, idx: usize) -> TokenStream2 {
    let ident = &f.ident;
    let name = kebab(&ident.to_string());
    let inner_ty = &f.inner_ty;
    let span = ident.span();
    let parse_one = quote_spanned! { span =>
        |s: &::std::string::String| -> ::std::result::Result<#inner_ty, ::std::string::String> {
            <#inner_ty as ::std::str::FromStr>::from_str(s)
                .map_err(|e| format!("<{}>: {}", #name, e))
        }
    };

    match f.shape {
        TypeShape::List if f.variadic => quote! {
            let #ident: ::std::vec::Vec<#inner_ty> = {
                let mut out: ::std::vec::Vec<#inner_ty> = ::std::vec::Vec::new();
                for s in &parsed.positionals[#idx..] {
                    out.push((#parse_one)(s)?);
                }
                out
            };
        },
        TypeShape::Opt => quote! {
            let #ident: ::std::option::Option<#inner_ty> = match parsed.positionals.get(#idx) {
                ::std::option::Option::Some(s) => ::std::option::Option::Some((#parse_one)(s)?),
                ::std::option::Option::None => ::std::option::Option::None,
            };
        },
        TypeShape::Scalar => {
            let default_tok = match &f.default {
                Some(v) => quote! {
                    {
                        let s = ::std::string::String::from(#v);
                        (#parse_one)(&s).expect("default value must parse")
                    }
                },
                None => quote! {
                    return ::std::result::Result::Err(
                        format!("missing required argument <{}>", #name)
                    )
                },
            };
            quote! {
                let #ident: #inner_ty = match parsed.positionals.get(#idx) {
                    ::std::option::Option::Some(s) => (#parse_one)(s)?,
                    ::std::option::Option::None => #default_tok,
                };
            }
        }
        _ => quote! {
            compile_error!("unsupported positional type shape");
        },
    }
}

/// `Option<&str>` as a token stream, so a missing description expands to
/// `None` rather than to an empty string that would render as a bare `-`.
fn opt_str(s: Option<&str>) -> TokenStream2 {
    match s {
        Some(v) => quote! { ::core::option::Option::Some(#v) },
        None => quote! { ::core::option::Option::None },
    }
}

/// The `[default: ...]` / `[possible: ...]` notes as a `[&str; N]` literal.
///
/// `flodl_cli::run` builds the same strings from a schema's JSON values; only
/// the layout is shared, because only the layout can be. Formatting them at
/// expand time keeps the generated help allocation-free apart from the
/// wrapping itself.
fn annotation_literals(default: Option<&str>, choices: Option<&[String]>) -> TokenStream2 {
    let mut items: Vec<String> = Vec::new();
    if let Some(d) = default {
        items.push(format!("[default: {d}]"));
    }
    if let Some(cs) = choices
        && !cs.is_empty()
    {
        items.push(format!("[possible: {}]", cs.join(", ")));
    }
    let lits = items
        .iter()
        .map(|s| quote! { ::std::string::String::from(#s) });
    quote! { [#(#lits),*] }
}

fn build_help_expr(
    fields: &[FieldSpec],
    description: Option<&str>,
    struct_name: &str,
) -> TokenStream2 {
    // Prefer the doc-comment description as the banner; fall back to the
    // struct ident only when no description is present. The struct name is
    // an implementation detail that users shouldn't see in `--help`.
    // Wrapped at runtime, like every row below: the banner carries the struct's
    // doc comment, which is the longest prose in the help and was the last part
    // still emitted as one unwrapped line per paragraph.
    let banner = description.unwrap_or(struct_name).to_string();

    // The help is assembled at runtime so `::flodl_cli::style::*` can check
    // whether stderr is a terminal — piped output stays plain, interactive
    // output gets ANSI color to match the hand-rolled helps in run.rs.
    // Padding is computed at macro-expand time from the raw label widths;
    // ANSI escapes are zero-width on terminal and don't affect alignment
    // because they're injected between the label and its trailing spaces.

    let mut arg_tokens: Vec<TokenStream2> = Vec::new();
    let mut opt_tokens: Vec<TokenStream2> = Vec::new();

    for f in fields {
        match f.kind {
            FieldKind::Option => {
                let long = kebab(&f.ident.to_string());
                let short_prefix = match f.short {
                    Some(c) => format!("-{c}, "),
                    None => String::from("    "),
                };
                let value_part = match f.shape {
                    TypeShape::Bool => String::new(),
                    TypeShape::List => String::from(" <VALUE>..."),
                    _ => format!(" <{}>", value_token(f)),
                };
                let label = format!("{short_prefix}--{long}{value_part}");
                let label_visible = label.chars().count();
                let desc = opt_str(f.description.as_deref());
                let anns = annotation_literals(f.default.as_deref(), f.choices.as_deref());
                opt_tokens.push(quote! {
                    {
                        let row = ::flodl_cli::help::row_string(
                            "    ",
                            &::flodl_cli::style::green(#label),
                            #label_visible,
                            ::flodl_cli::help::DERIVE_COL,
                            #desc,
                            &#anns,
                        );
                        // Separate an entry whose description wrapped, so its
                        // last line cannot read as the next entry's first.
                        let tall = row.matches('\n').count() > 1;
                        if !first && (prev_tall || tall) {
                            out.push('\n');
                        }
                        out.push_str(&row);
                        prev_tall = tall;
                        first = false;
                    }
                });
            }
            FieldKind::Arg => {
                let name = kebab(&f.ident.to_string());
                let required = matches!(f.shape, TypeShape::Scalar) && f.default.is_none();
                let label = if f.variadic {
                    format!("<{name}>...")
                } else if required {
                    format!("<{name}>")
                } else {
                    format!("[<{name}>]")
                };
                let label_visible = label.chars().count();
                let desc = opt_str(f.description.as_deref());
                let anns = annotation_literals(f.default.as_deref(), f.choices.as_deref());
                arg_tokens.push(quote! {
                    {
                        let row = ::flodl_cli::help::row_string(
                            "    ",
                            &::flodl_cli::style::green(#label),
                            #label_visible,
                            ::flodl_cli::help::DERIVE_COL,
                            #desc,
                            &#anns,
                        );
                        // Separate an entry whose description wrapped, so its
                        // last line cannot read as the next entry's first.
                        let tall = row.matches('\n').count() > 1;
                        if !first && (prev_tall || tall) {
                            out.push('\n');
                        }
                        out.push_str(&row);
                        prev_tall = tall;
                        first = false;
                    }
                });
            }
        }
    }

    let arg_section = if arg_tokens.is_empty() {
        quote! {}
    } else {
        quote! {
            out.push_str(&::flodl_cli::style::yellow("Arguments"));
            out.push_str(":\n");
            let mut first = true;
            let mut prev_tall = false;
            #(#arg_tokens)*
            out.push('\n');
        }
    };
    let opt_section = if opt_tokens.is_empty() {
        quote! {}
    } else {
        quote! {
            out.push_str(&::flodl_cli::style::yellow("Options"));
            out.push_str(":\n");
            let mut first = true;
            let mut prev_tall = false;
            #(#opt_tokens)*
            out.push('\n');
        }
    };

    quote! {
        {
            let mut out = ::flodl_cli::help::banner_string(#banner);
            #arg_section
            #opt_section
            out
        }
    }
}

fn value_token(f: &FieldSpec) -> &'static str {
    let inner = inner_ty_name(&f.inner_ty);
    match inner.as_str() {
        "u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize" => "N",
        "f32" | "f64" => "F",
        "PathBuf" | "Path" => "PATH",
        _ => "VALUE",
    }
}

// ── Utilities ───────────────────────────────────────────────────────────

/// Collect a doc comment into the block form the help renderers consume.
///
/// Blocks are separated by `\n`; a block that begins with `- ` is a list
/// item. Within a block, source line breaks are irrelevant (the renderer
/// re-wraps to the terminal) so they collapse to single spaces.
///
/// This used to be `lines.join(" ").split_whitespace().join(" ")`, which
/// flattened the whole comment into one paragraph. The structure was
/// therefore destroyed HERE, before any renderer could honor it — which is
/// why a doc comment carrying a perfectly good bullet list rendered as a
/// run-on sentence with inline dashes ("Accepts: - `rank-N`: ... - `fastest`:
/// ..."). Preserving it is only half the fix; see
/// `flodl_cli::help::wrap_description` for the other half.
///
/// A block break is either a blank line or the start of a new bullet, so a
/// list item spanning several source lines stays ONE item. That is also why
/// the per-line `trim()` is safe to keep: continuation lines are recognised
/// by NOT starting a bullet, never by their indent.
fn extract_doc(attrs: &[Attribute]) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    for a in attrs {
        if !a.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &a.meta
            && let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
        {
            let text = s.value();
            lines.push(text.trim().to_string());
        }
    }
    doc_blocks(&lines)
}

/// The string half of [`extract_doc`], split out so it is testable without
/// hand-building `syn::Attribute` values.
fn doc_blocks(lines: &[String]) -> Option<String> {
    if lines.is_empty() {
        return None;
    }

    let mut blocks: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in lines {
        if line.is_empty() {
            if !current.is_empty() {
                blocks.push(std::mem::take(&mut current));
            }
            continue;
        }
        // `*` is accepted as a bullet marker on input and normalised to `-`,
        // so the renderer has one shape to detect. A bare `-`/`*` with no
        // following space is a word (a negative number, a glob), not a
        // bullet.
        let bullet = line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .map(str::trim_start);
        match bullet {
            Some(rest) => {
                if !current.is_empty() {
                    blocks.push(std::mem::take(&mut current));
                }
                current.push_str("- ");
                current.push_str(rest);
            }
            None => {
                // A blank line already flushed `current`, so a non-empty
                // buffer here always means "continuation of this block".
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(line);
            }
        }
    }
    if !current.is_empty() {
        blocks.push(current);
    }

    // Collapse whitespace runs inside each block, drop empties, then join
    // with `\n` so the block boundaries survive into the JSON schema.
    let joined = blocks
        .iter()
        .map(|b| b.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|b| !b.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

fn kebab(s: &str) -> String {
    s.replace('_', "-")
}

/// PascalCase enum variant ident → kebab-case subcommand name.
/// `Train` → `train`, `TrainSubscan` → `train-subscan`,
/// `EvalLetterDirect` → `eval-letter-direct`. A leading capital does not
/// get a separator; underscores are also treated as boundaries.
fn pascal_to_kebab(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if c == '_' {
            out.push('-');
        } else if c.is_uppercase() {
            if i != 0 && !out.ends_with('-') {
                out.push('-');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

// syn's Span import trick: pull from proc_macro2 above.
use syn::spanned::Spanned;

#[cfg(test)]
mod tests {
    use super::{doc_blocks, pascal_to_kebab};

    fn doc(src: &[&str]) -> Option<String> {
        let lines: Vec<String> = src.iter().map(|s| s.trim().to_string()).collect();
        doc_blocks(&lines)
    }

    /// The regression this whole change exists for: a bullet list in a doc
    /// comment came out as a run-on sentence with inline dashes, because the
    /// structure was destroyed here rather than at the renderer.
    #[test]
    fn doc_blocks_keeps_a_bullet_list_apart() {
        let out = doc(&[
            "Which rank fires per-epoch user callbacks (`epoch_fn`,",
            "`checkpoint_fn`, `eval_fn`). Accepts:",
            "",
            "- `rank-N`: explicit rank index (e.g. `rank-0`, `rank-1`).",
            "  Loud-errors at framework validation if `N >= world_size`.",
            "- `fastest`: ElChe picks the rank with the lowest",
            "  smoothed-ms-per-batch at run start, sticky thereafter.",
            "",
            "Default: framework default (`Rank(0)`). Solo modes ignore this.",
        ])
        .expect("some");

        let blocks: Vec<&str> = out.split('\n').collect();
        assert_eq!(
            blocks.len(),
            4,
            "lead paragraph + 2 items + trailer: {out:?}"
        );
        assert!(blocks[0].ends_with("Accepts:"), "{:?}", blocks[0]);
        assert!(blocks[1].starts_with("- `rank-N`:"), "{:?}", blocks[1]);
        assert!(
            blocks[1].ends_with("if `N >= world_size`."),
            "a multi-line item must stay ONE block: {:?}",
            blocks[1]
        );
        assert!(blocks[2].starts_with("- `fastest`:"), "{:?}", blocks[2]);
        assert!(blocks[3].starts_with("Default:"), "{:?}", blocks[3]);
    }

    /// A new bullet ends the previous block even with no blank line between
    /// items — the common way lists are actually written.
    #[test]
    fn doc_blocks_splits_adjacent_bullets() {
        let out = doc(&["Modes:", "- one", "- two", "- three"]).expect("some");
        assert_eq!(out, "Modes:\n- one\n- two\n- three");
    }

    /// `*` is accepted on input and normalised, so the renderer detects one
    /// marker shape.
    #[test]
    fn doc_blocks_normalises_star_bullets() {
        let out = doc(&["* alpha", "* beta"]).expect("some");
        assert_eq!(out, "- alpha\n- beta");
    }

    /// A lone `-`/`*` with no following space is a word, not a bullet: a
    /// negative number or a glob must not split a paragraph.
    #[test]
    fn doc_blocks_does_not_treat_bare_dash_as_a_bullet() {
        let out = doc(&["Range is", "-1.0 to 1.0 and", "*.rs is a glob"]).expect("some");
        assert_eq!(out, "Range is -1.0 to 1.0 and *.rs is a glob");
    }

    /// The ordinary case must be unchanged: a plain multi-line comment is
    /// still one reflowable paragraph.
    #[test]
    fn doc_blocks_folds_a_plain_comment_into_one_paragraph() {
        let out = doc(&["Override epoch", "count.", ""]).expect("some");
        assert_eq!(out, "Override epoch count.");
        assert!(doc(&[]).is_none());
        assert!(doc(&["", "  ", ""]).is_none());
    }

    #[test]
    fn pascal_to_kebab_maps_variant_idents() {
        assert_eq!(pascal_to_kebab("Train"), "train");
        assert_eq!(pascal_to_kebab("Eval"), "eval");
        // The fbrl `word` modes — the generalization-stressing cases.
        assert_eq!(pascal_to_kebab("TrainSubscan"), "train-subscan");
        assert_eq!(pascal_to_kebab("EvalSubscan"), "eval-subscan");
        assert_eq!(pascal_to_kebab("EvalLetterDirect"), "eval-letter-direct");
        // Underscores are boundaries too; no double separators.
        assert_eq!(pascal_to_kebab("Train_Subscan"), "train-subscan");
        assert_eq!(pascal_to_kebab("Generate"), "generate");
    }
}
