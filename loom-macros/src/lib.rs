//! Procedural macros for loom-rs runtime.
//!
//! This crate provides the `#[loom_rs::test]` attribute macro for writing
//! tests that run within a LoomRuntime.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, ItemFn, Meta};

/// Configuration parsed from the macro attributes.
#[derive(Default)]
struct TestConfig {
    tokio_thread_count: Option<usize>,
    rayon_thread_count: Option<usize>,
}

impl TestConfig {
    fn parse(attrs: &[Meta]) -> syn::Result<Self> {
        let mut config = Self::default();

        for meta in attrs {
            if let Meta::NameValue(nv) = meta {
                let ident = nv
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new_spanned(&nv.path, "expected identifier"))?;

                let value = match &nv.value {
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(lit),
                        ..
                    }) => lit.base10_parse::<usize>()?,
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &nv.value,
                            "expected integer literal",
                        ))
                    }
                };

                match ident.to_string().as_str() {
                    "tokio_thread_count" => config.tokio_thread_count = Some(value),
                    "rayon_thread_count" => config.rayon_thread_count = Some(value),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            ident,
                            format!(
                                "unknown attribute `{}`, expected `tokio_thread_count` or `rayon_thread_count`",
                                ident
                            ),
                        ))
                    }
                }
            } else {
                return Err(syn::Error::new_spanned(
                    meta,
                    "expected `key = value` format",
                ));
            }
        }

        Ok(config)
    }
}

/// A test attribute macro for loom-rs that sets up a LoomRuntime with test-appropriate defaults.
///
/// # Default Configuration
///
/// - 1 tokio thread
/// - 2 rayon threads
/// - Thread pinning disabled
///
/// # Attributes
///
/// - `tokio_thread_count = N` - Set the number of tokio worker threads
/// - `rayon_thread_count = N` - Set the number of rayon threads
///
/// # Examples
///
/// Basic usage with defaults:
///
/// ```ignore
/// #[loom_rs::test]
/// async fn test_spawn_compute() {
///     let result = loom_rs::spawn_compute(|| 42).await;
///     assert_eq!(result, 42);
/// }
/// ```
///
/// Custom thread counts:
///
/// ```ignore
/// #[loom_rs::test(tokio_thread_count = 2, rayon_thread_count = 4)]
/// async fn test_parallel_work() {
///     // Test code here
/// }
/// ```
#[proc_macro_attribute]
pub fn test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    // Parse attributes
    let attr_parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
    let attrs = match syn::parse::Parser::parse(attr_parser, attr) {
        Ok(attrs) => attrs,
        Err(e) => return e.to_compile_error().into(),
    };

    let config = match TestConfig::parse(&attrs.into_iter().collect::<Vec<_>>()) {
        Ok(c) => c,
        Err(e) => return e.to_compile_error().into(),
    };

    match generate_test(input, config) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generate_test(input: ItemFn, config: TestConfig) -> syn::Result<TokenStream2> {
    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = input;

    // Verify the function is async
    if sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            sig.fn_token,
            "test function must be async",
        ));
    }

    let fn_name = &sig.ident;

    // Get thread counts with defaults
    let tokio_threads = config.tokio_thread_count.unwrap_or(1);
    let rayon_threads = config.rayon_thread_count.unwrap_or(2);

    // Create the new synchronous function signature
    let mut new_sig = sig.clone();
    new_sig.asyncness = None;

    // Generate the test function
    let output = quote! {
        #[::core::prelude::v1::test]
        #(#attrs)*
        #vis #new_sig {
            let __loom_runtime = ::loom_rs::LoomBuilder::new()
                .prefix(concat!("test-", stringify!(#fn_name)))
                .tokio_threads(#tokio_threads)
                .rayon_threads(#rayon_threads)
                .pin_threads(false)
                .build()
                .expect("failed to create test runtime");

            __loom_runtime.block_on(async #block);
            __loom_runtime.block_until_idle();
        }
    };

    Ok(output)
}
