#![feature(if_let)]
#![feature(plugin_registrar)]
#![feature(slicing_syntax)]
#![feature(tuple_indexing)]
#![feature(while_let)]

extern crate rustc;
extern crate syntax;

use syntax::ast::{mod, TokenTree};
use syntax::ext::base::{DummyResult, ExtCtxt, MacResult, MacExpr};
use syntax::ext::quote::rt::ToSource;
use syntax::codemap::Span;
use syntax::parse;
use syntax::ptr::P;

#[doc(hidden)]
#[plugin_registrar]
pub fn registrar(registry: &mut rustc::plugin::Registry) {
    registry.register_macro("to_glsl", expand);
}

/// Expand to_glsl!
fn expand(ecx: &mut ExtCtxt, span: Span, input: &[TokenTree]) -> Box<MacResult + 'static> {
    use syntax::parse::token;
    use syntax::ext::build::AstBuilder;

    let glsl = match rust_to_glsl(input, &GlslVersion(1, 5)) {
        Ok(glsl) => glsl,
        Err(e) => {
            ecx.span_err(span, format!("{}", e)[]);
            return DummyResult::any(span);
        }
    };

    let glsl = token::intern_and_get_ident(glsl[]);
    MacExpr::new(ecx.expr_lit(span, ast::LitStr(glsl, ast::CookedStr)))
}

/// Represents a version of GLSL.
pub struct GlslVersion(pub u8, pub u8);

/// Error that can happen while converting.
#[deriving(Show, Clone)]
pub enum Error {
    /// Found a type that is not supported 
    UnsupportedType(String),
    /// Found generics.
    ///
    /// Contains the source code of the item containing them.
    FoundGenerics(String),
    /// Found a Rust construct that is not supported by GLSL, like a `mod` or a `trait`.
    ///
    /// Contains the source code of the unexpected item or expr.
    UnexpectedConstruct(String),
    Misc,        // TODO: remove this error type
}

/// Turns Rust code into GLSL.
pub fn rust_to_glsl(input: &[TokenTree], max_glsl_version: &GlslVersion) -> Result<String, Error> {
    let mut result = String::new();

    // this variable contains the minimum required GLSL version
    let mut req_glsl_version = GlslVersion(1, 1);

    let parse_sess = parse::new_parse_sess();
    let mut parser = parse::new_parser_from_tts(&parse_sess, Vec::new(), input.to_vec());

    while let Some(item) = parser.parse_item_with_outer_attributes() {
        let item_name = item.ident.as_str();

        match item.node {
            ast::ItemFn(ref decl, _, _, ref generics, ref block) => {
                if generics.lifetimes.len() != 0 || generics.ty_params.len() != 0 ||
                    generics.where_clause.predicates.len() != 0
                {
                    return Err(FoundGenerics(item.to_source()))
                }

                // TODO: args
                result.push_str(format!("{ret} {name}() {{\n",
                    name = item_name, ret = try!(ty_to_glsl(&decl.output)))[]);

                result.push_str(try!(block_to_glsl(block))[]);

                result.push_str("}");
            },

            ast::ItemStatic(ref ty, _, _) => {
                result.push_str(format!("uniform {ty} {name};\n",
                    ty = try!(ty_to_glsl(ty)), name = item_name)[]);
            },

            ast::ItemConst(ref ty, ref expr) => {
                result.push_str(format!("const {ty} {name} = {expr};\n",
                    ty = try!(ty_to_glsl(ty)), name = item_name, expr = expr.to_source())[]);
            },

            ast::ItemMod(..) | ast::ItemForeignMod(..) | ast::ItemTrait(..) | ast::ItemImpl(..) |
            ast::ItemMac(..) => {
                return Err(UnexpectedConstruct(item.to_source()))
            }

            _ => unimplemented!()
        }
    }

    if !parser.eat(&parse::token::Eof) {
        return Err(Misc);       // TODO: 
    }

    // prepending req_glsl_version
    let result = format!("#version {}{}0\n{}", req_glsl_version.0, req_glsl_version.1, result);
    Ok(result)
}

/// Turns a Ty into a GLSL type.
fn ty_to_glsl(ty: &P<ast::Ty>) -> Result<String, Error> {
    let ty = ty.to_source();

    let overwrite = match ty[] {
        "()" => Some("void"),
        "int" => Some("int"),
        "uint" => Some("unsigned int"),
        "i8" => Some("char"),
        "u8" => Some("unsigned char"),
        "i16" => Some("short"),
        "u16" => Some("unsigned short"),
        "i32" => Some("int"),
        "u32" => Some("unsigned int"),
        "f32" => Some("float"),
        "f64" => return Err(UnsupportedType("f64".to_string())),
        _ => None
    };

    Ok(match overwrite {
        Some(r) => r.to_string(),
        None => ty
    })
}

/// Turns a Stmt into a GLSL expression.
fn stmt_to_glsl(stmt: &P<ast::Stmt>) -> Result<String, Error> {
    match stmt.node {
        ast::StmtDecl(ref decl, _) => unimplemented!(),
        ast::StmtExpr(ref expr, _) => expr_to_glsl(expr),
        ast::StmtSemi(ref expr, _) => expr_to_glsl(expr),
        ast::StmtMac(..) => Err(UnexpectedConstruct(stmt.to_source()))
    }
}

/// Turns a Stmt into a GLSL expression.
fn block_to_glsl(block: &P<ast::Block>) -> Result<String, Error> {
    let mut result = String::new();

    for stmt in block.stmts.iter() {
        result.push_str(try!(stmt_to_glsl(stmt))[]);
        result.push_str("\n");
    }

    if let Some(ref expr) = block.expr {
        result.push_str("return ");
        result.push_str(try!(expr_to_glsl(expr))[]);
        result.push_str(";\n");
    }

    Ok(result)
}

/// Turns an Expr into a GLSL expression.
fn expr_to_glsl(expr: &P<ast::Expr>) -> Result<String, Error> {
    match expr.node {
        ast::ExprCast(ref e, ref t) => Ok(format!("(({t})({e}))", t=t, e=e)),
        ast::ExprLit(ref lit) => lit_to_glsl(lit),
        ast::ExprRet(Some(ref expr)) => Ok(format!("return {e};", e = try!(expr_to_glsl(expr)))),
        ast::ExprRet(None) => Ok(format!("return;")),
        /*ExprVec(Vec<P<Expr>>),
        ExprCall(P<Expr>, Vec<P<Expr>>),
        ExprMethodCall(SpannedIdent, Vec<P<Ty>>, Vec<P<Expr>>),
        ExprTup(Vec<P<Expr>>),
        ExprBinary(BinOp, P<Expr>, P<Expr>),
        ExprUnary(UnOp, P<Expr>),
        ExprIf(P<Expr>, P<Block>, Option<P<Expr>>),
        ExprWhile(P<Expr>, P<Block>, Option<Ident>),
        ExprForLoop(P<Pat>, P<Expr>, P<Block>, Option<Ident>),
        ExprLoop(P<Block>, Option<Ident>),
        ExprMatch(P<Expr>, Vec<Arm>, MatchSource),
        ExprFnBlock(CaptureClause, P<FnDecl>, P<Block>),
        ExprProc(P<FnDecl>, P<Block>),
        ExprUnboxedFn(CaptureClause, UnboxedClosureKind, P<FnDecl>, P<Block>),
        ExprBlock(P<Block>),
        ExprAssign(P<Expr>, P<Expr>),
        ExprAssignOp(BinOp, P<Expr>, P<Expr>),
        ExprField(P<Expr>, SpannedIdent, Vec<P<Ty>>),
        ExprTupField(P<Expr>, Spanned<uint>, Vec<P<Ty>>),
        ExprIndex(P<Expr>, P<Expr>),
        ExprSlice(P<Expr>, Option<P<Expr>>, Option<P<Expr>>, Mutability),
        ExprPath(Path),
        ExprAddrOf(Mutability, P<Expr>),
        ExprBreak(Option<Ident>),
        ExprAgain(Option<Ident>),
        ExprRet(Option<P<Expr>>),
        ExprInlineAsm(InlineAsm),
        ExprMac(Mac),
        ExprStruct(Path, Vec<Field>, Option<P<Expr>>),
        ExprRepeat(P<Expr>, P<Expr>),
        ExprParen(P<Expr>),*/

        _ => Err(UnexpectedConstruct(expr.to_source())),
    }
}

/// Turns a literal into a GLSL expression.
fn lit_to_glsl(lit: &P<ast::Lit>) -> Result<String, Error> {
    match lit.node {
        ast::LitInt(val, ast::SignedIntLit(_, ast::Minus)) => Ok(format!("-{}", val)),
        ast::LitInt(val, _) => Ok(format!("{}", val)),
        ast::LitFloat(ref s, _) => Ok(s.get().to_string()),
        ast::LitFloatUnsuffixed(ref s) => Ok(s.get().to_string()),
        ast::LitBool(val) => Ok(if val { "true" } else { "false" }.to_string()),

        ast::LitStr(ref val, _) => Err(UnexpectedConstruct(val.get().to_string())),
        ast::LitBinary(ref val) => Err(UnexpectedConstruct(val.to_string())),
        ast::LitByte(val) => Err(UnexpectedConstruct(val.to_string())),
        ast::LitChar(val) => Err(UnexpectedConstruct(val.to_string())),
        ast::LitNil => Err(UnexpectedConstruct("()".to_string())),
    }
}
