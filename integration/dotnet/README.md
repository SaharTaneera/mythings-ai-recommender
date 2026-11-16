# Example .NET integration

This folder is a **contract-level C# example** showing how an existing .NET 8+ ASP.NET Core backend could call the FastAPI recommender. It is not an actual integrated MyThings .NET service.

The `MyThingsRecommendationClient.cs` file supplies request/response DTOs with explicit JSON property names matching the Python Pydantic schemas. It uses a typed `HttpClient` and cancellation propagation. It requires no extra NuGet packages.

## Registration in an existing ASP.NET Core `Program.cs`

```csharp
using MyThings.Recommendations;

builder.Services.AddHttpClient<MyThingsRecommendationClient>(client =>
{
    client.BaseAddress = new Uri(
        builder.Configuration["RecommendationApi:BaseUrl"]
        ?? "http://127.0.0.1:8000/");
    client.Timeout = TimeSpan.FromSeconds(120);
});
```

Configure `RecommendationApi:BaseUrl` for the deployment. `localhost` only works when the .NET process can reach the API on the same host. The trailing slash in `BaseAddress` ensures the relative request path resolves as expected.

## Example request

```csharp
var cart = new CartDto("R001", new[] { new CartItemDto("101", 1) });
var cartProducts = new[]
{
    new ProductDto("101", "R001", "Chicken Shawarma", "شاورما دجاج", "main_dish")
};
var candidates = new[]
{
    new ProductDto("102", "R001", "Garlic Sauce", "ثومية", "sauce"),
    new ProductDto("103", "R001", "French Fries", "بطاطا مقلية", "side_dish")
};

var result = await recommendationClient.RecommendAsync(
    new RecommendationRequestDto(cart, cartProducts, candidates, TopK: 3),
    cancellationToken);
```

This snippet assumes `recommendationClient` is injected into a service/controller and `cancellationToken` is available.

## Critical ownership boundaries

The commerce backend—not the model—must authenticate/authorize user and merchant access, fetch trusted product metadata, filter inventory and seller restrictions, impose candidate limits, validate returned IDs and current prices, and implement timeouts/fallbacks. The model scores are **heuristic ranking signals**, not calibrated purchase probabilities. Never accept an untrusted browser-provided merchant or product pool as authorization. The prototype API has no built-in authentication.
