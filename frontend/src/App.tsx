
import { useEffect, useState } from "react";
import "./App.css";

type Product = {
  id: string;
  merchant_id: string;
  name_en: string;
  name_ar: string;
  category?: string;
  description_en?: string;
  price: number; // Demo display data only
  emoji: string;
};

type CartItem = {
  product_id: string;
  quantity: number;
};

type RecommendationResponse = {
  recommendations: {
    product_id: string;
    score: number;
  }[];
  model_version: string;
};

const MERCHANT_ID = "R001";

const catalog: Product[] = [
  {
    id: "101",
    merchant_id: MERCHANT_ID,
    name_en: "Chicken Shawarma",
    name_ar: "شاورما دجاج",
    category: "main_dish",
    price: 2.5,
    emoji: "🌯",
  },
  {
    id: "102",
    merchant_id: MERCHANT_ID,
    name_en: "Garlic Sauce",
    name_ar: "ثومية",
    category: "sauce",
    description_en: "Savory garlic dipping sauce for shawarma",
    price: 0.5,
    emoji: "🧄",
  },
  {
    id: "103",
    merchant_id: MERCHANT_ID,
    name_en: "Pepsi",
    name_ar: "بيبسي",
    category: "beverage",
    price: 0.75,
    emoji: "🥤",
  },
  {
    id: "104",
    merchant_id: MERCHANT_ID,
    name_en: "French Fries",
    name_ar: "بطاطا مقلية",
    category: "side_dish",
    price: 1.25,
    emoji: "🍟",
  },
  {
    id: "105",
    merchant_id: MERCHANT_ID,
    name_en: "Beef Burger",
    name_ar: "برجر لحم",
    category: "main_dish",
    price: 3.5,
    emoji: "🍔",
  },
  {
    id: "106",
    merchant_id: MERCHANT_ID,
    name_en: "Chocolate Sauce",
    name_ar: "صوص شوكولاتة",
    category: "sauce",
    description_en: "Sweet chocolate topping for desserts",
    price: 0.6,
    emoji: "🍫",
  },
];

function toAiProduct(product: Product) {
  return {
    id: product.id,
    merchant_id: product.merchant_id,
    name_en: product.name_en,
    name_ar: product.name_ar,
    category: product.category,
    description_en: product.description_en,
  };
}

function App() {
  const [language, setLanguage] = useState<"en" | "ar">("en");
  const [cart, setCart] = useState<CartItem[]>([
    { product_id: "101", quantity: 1 },
  ]);
  const [recommendations, setRecommendations] =
    useState<RecommendationResponse["recommendations"]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelVersion, setModelVersion] = useState("");

  const arabic = language === "ar";

  function name(product: Product) {
    return arabic ? product.name_ar : product.name_en;
  }

  function addToCart(productId: string) {
    setCart((current) => {
      const existing = current.find(
        (item) => item.product_id === productId
      );

      if (existing) {
        return current.map((item) =>
          item.product_id === productId
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }

      return [...current, { product_id: productId, quantity: 1 }];
    });
  }

  function removeFromCart(productId: string) {
    setCart((current) =>
      current.filter((item) => item.product_id !== productId)
    );
  }

  useEffect(() => {
    const controller = new AbortController();

    async function fetchRecommendations() {
      if (cart.length === 0) {
        setRecommendations([]);
        setModelVersion("");
        setError("");
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");
      setRecommendations([]);

      const cartIds = new Set(
        cart.map((item) => item.product_id)
      );

      const cartProducts = catalog.filter(
        (product) => cartIds.has(product.id)
      );

      const candidates = catalog.filter(
        (product) =>
          product.merchant_id === MERCHANT_ID &&
          !cartIds.has(product.id)
      );

      try {
        const response = await fetch("/api/v1/recommendations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            cart: {
              merchant_id: MERCHANT_ID,
              items: cart,
            },
            products: cartProducts.map(toAiProduct),
            candidates: candidates.map(toAiProduct),
            top_k: 3,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data: RecommendationResponse =
          await response.json();

        if (!controller.signal.aborted) {
          setRecommendations(data.recommendations);
          setModelVersion(data.model_version);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "Could not load recommendations"
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    fetchRecommendations();

    return () => controller.abort();
  }, [cart]);

  const total = cart.reduce((sum, item) => {
    const product = catalog.find(
      (p) => p.id === item.product_id
    );
    return sum + (product?.price ?? 0) * item.quantity;
  }, 0);

  return (
    <div className="app" dir={arabic ? "rtl" : "ltr"}>
      <header className="header">
        <div>
          <span className="brand">mythings.</span>
          <p className="subtitle">AI Recommendation Demo</p>
        </div>

        <button
          className="language-button"
          onClick={() =>
            setLanguage(arabic ? "en" : "ar")
          }
        >
          {arabic ? "English" : "العربية"}
        </button>
      </header>

      <main className="layout">
        <section className="panel">
          <div className="section-header">
            <div>
              <h2>{arabic ? "قائمة المطعم" : "Restaurant Menu"}</h2>
              <p className="muted">
                {arabic ? "مطعم تجريبي R001" : "Demo Restaurant R001"}
              </p>
            </div>
          </div>

          <div className="product-grid">
            {catalog.map((product) => (
              <div className="product-card" key={product.id}>
                <div className="product-emoji">{product.emoji}</div>
                <h3>{name(product)}</h3>
                <p className="price">
                  {product.price.toFixed(2)} JOD
                </p>
                <button
                  className="primary-button"
                  onClick={() => addToCart(product.id)}
                >
                  {arabic ? "أضف للسلة +" : "+ Add to cart"}
                </button>
              </div>
            ))}
          </div>
        </section>

        <aside className="sidebar">
          <section className="panel">
            <h2>{arabic ? "سلة التسوق" : "Your Cart"}</h2>

            {cart.length === 0 ? (
              <p className="muted">
                {arabic ? "السلة فارغة" : "Your cart is empty"}
              </p>
            ) : (
              cart.map((item) => {
                const product = catalog.find(
                  (p) => p.id === item.product_id
                );

                if (!product) return null;

                return (
                  <div className="cart-item" key={product.id}>
                    <span className="item-emoji">
                      {product.emoji}
                    </span>
                    <div className="cart-details">
                      <strong>{name(product)}</strong>
                      <p className="muted">
                        {arabic ? "الكمية" : "Qty"}: {item.quantity}
                      </p>
                    </div>
                    <button
                      className="remove-button"
                      onClick={() => removeFromCart(product.id)}
                      aria-label={`Remove ${product.name_en}`}
                    >
                      ×
                    </button>
                  </div>
                );
              })
            )}

            <div className="total-row">
              <strong>{arabic ? "المجموع" : "Total"}</strong>
              <strong>{total.toFixed(2)} JOD</strong>
            </div>
          </section>

          <section className="panel recommendations">
            <div className="recommendation-title">
              <span>✨</span>
              <h2>
                {arabic
                  ? "أكمل طلبك"
                  : "Complete Your Order"}
              </h2>
            </div>

            <p className="muted">
              {arabic
                ? "اقتراحات من نموذج الذكاء الاصطناعي"
                : "Suggestions from our AI model"}
            </p>

            {loading && (
              <p className="muted">
                {arabic ? "جاري التحميل..." : "Finding recommendations..."}
              </p>
            )}

            {error && <p className="error">{error}</p>}

            {!loading && !error &&
              recommendations.length === 0 && (
                <p className="muted">
                  {arabic
                    ? "لا توجد اقتراحات حالياً"
                    : "No recommendations available"}
                </p>
              )}

            {!loading && recommendations.map((recommendation) => {
              const product = catalog.find(
                (p) => p.id === recommendation.product_id
              );

              if (!product) return null;

              return (
                <div className="recommendation-item" key={product.id}>
                  <span className="item-emoji">
                    {product.emoji}
                  </span>

                  <div className="cart-details">
                    <strong>{name(product)}</strong>
                    <p className="muted">
                      {product.price.toFixed(2)} JOD
                    </p>
                  </div>

                  <button
                    className="add-small"
                    onClick={() => addToCart(product.id)}
                    aria-label={`Add ${product.name_en}`}
                  >
                    +
                  </button>
                </div>
              );
            })}

            {modelVersion && (
              <p className="model-label">
                Model: {modelVersion}
              </p>
            )}
          </section>
        </aside>
      </main>

      <footer className="footer">
        Demonstration using synthetic catalog and prices.
        Recommendations are generated by the local AI service.
      </footer>
    </div>
  );
}

export default App;
